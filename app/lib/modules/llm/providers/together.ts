// ~/lib/modules/llm/providers/together-provider.ts
import { BaseProvider, getOpenAILikeModel } from '~/lib/modules/llm/base-provider';
import type { ModelInfo } from '~/lib/modules/llm/types';
import type { IProviderSetting } from '~/types/model';
import type { LanguageModelV1 } from 'ai';

type ServerEnv = Record<string, string>;

/**
 * AgentRouter provider wrapper
 */
export default class TogetherProvider extends BaseProvider {
  // provider name (use a plain string here to be safe in field initializers)
  name = 'AgentRouter';
  getApiKeyLink = 'https://agentrouter.org/';

  config = {
    baseUrlKey: 'ANTHROPIC_BASE_URL',
    apiTokenKey: 'ANTHROPIC_API_KEY',
  };

  // Keep a static model example (provider set to the provider name)
  staticModels: ModelInfo[] = [
    {
      name: 'claude-sonnet-4-5-20250929',
      label: 'Claude Sonnet 4.5',
      provider: 'AgentRouter',
      maxTokenAllowed: 200000,
      maxCompletionTokens: 8192,
    },
  ];

  /**
   * Fetch dynamic models from AgentRouter (v1beta /v1 depending on API)
   */
  async getDynamicModels(
    apiKeys?: Record<string, string>,
    settings?: IProviderSetting,
    serverEnv: ServerEnv = {},
  ): Promise<ModelInfo[]> {
    const { baseUrl: fetchBaseUrl, apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: settings,
      serverEnv,
      defaultBaseUrlKey: 'ANTHROPIC_BASE_URL',
      defaultApiTokenKey: 'ANTHROPIC_API_KEY',
    });

    // Fallback sensible defaults
    const baseUrl = (fetchBaseUrl && String(fetchBaseUrl).trim()) || serverEnv.ANTHROPIC_BASE_URL || 'https://agentrouter.org/';
    const key = apiKey || serverEnv.ANTHROPIC_API_KEY || serverEnv.AGENTROUTER_API_KEY || apiKeys?.[this.name] || apiKeys?.ANTHROPIC_API_KEY;

    if (!baseUrl || !key) {
      // No credentials -> return static models only (or empty list)
      return this.staticModels || [];
    }

    // Try multiple candidate endpoints and header styles (Authorization Bearer, x-api-key)
    const endpoints = [
      `${baseUrl.replace(/\/$/, '')}/v1beta/models`,
      `${baseUrl.replace(/\/$/, '')}/v1/models`,
      `${baseUrl.replace(/\/$/, '')}/models`,
    ];

    let resJson: any = null;
    for (const url of endpoints) {
      try {
        const resp = await fetch(url, {
          headers: {
            // primary common header
            Authorization: `Bearer ${key}`,
            Accept: 'application/json',
          },
        });

        // if 401/403, try alternate header style once
        if (resp.status === 401 || resp.status === 403) {
          // try x-api-key style
          const alt = await fetch(url, {
            headers: { 'x-api-key': key, Accept: 'application/json' },
          });
          if (alt.ok) {
            try {
              resJson = await alt.json();
              break;
            } catch {
              continue;
            }
          } else {
            continue;
          }
        }

        if (!resp.ok) {
          // not ok -> continue to next endpoint
          continue;
        }

        try {
          resJson = await resp.json();
          break;
        } catch (e) {
          // parse error -> continue
          continue;
        }
      } catch (e) {
        // network error -> try next endpoint
        continue;
      }
    }

    if (!resJson) {
      // nothing fetched — return static models to avoid empty UX
      return this.staticModels || [];
    }

    // Normalize response shapes:
    // possible shapes: Array, { models: [...] }, { data: [...] }, { result: [...] }, { items: [...] }
    let rawModels: any[] = [];
    if (Array.isArray(resJson)) {
      rawModels = resJson;
    } else if (Array.isArray(resJson.models)) {
      rawModels = resJson.models;
    } else if (Array.isArray(resJson.data)) {
      rawModels = resJson.data;
    } else if (Array.isArray(resJson.result)) {
      rawModels = resJson.result;
    } else if (Array.isArray(resJson.items)) {
      rawModels = resJson.items;
    } else {
      // try to find an array anywhere in the object
      const maybeArray = Object.values(resJson).find((v) => Array.isArray(v));
      if (Array.isArray(maybeArray)) rawModels = maybeArray as any[];
    }

    if (!Array.isArray(rawModels) || rawModels.length === 0) {
      return this.staticModels || [];
    }

    // Filter chat-like models defensively
    const chatModels = rawModels.filter((m: any) => {
      if (!m) return false;
      if (typeof m.type === 'string') return m.type.toLowerCase() === 'chat';
      const id = String(m.id || m.name || '').toLowerCase();
      return id.includes('chat') || id.includes('claude') || id.includes('sonnet') || id.includes('assistant');
    });

    const mapped: ModelInfo[] = chatModels.map((m: any) => {
      const id = m.id || m.model_id || m.name || (m._id && String(m._id));
      const display = m.display_name || m.name || id || 'unknown-model';

      let pricePart = '';
      try {
        if (m.pricing && typeof m.pricing.input === 'number' && typeof m.pricing.output === 'number') {
          pricePart = ` - in:$${m.pricing.input.toFixed(2)} out:$${m.pricing.output.toFixed(2)}`;
        } else if (typeof m.price === 'number') {
          pricePart = ` - $${m.price.toFixed(4)}`;
        }
      } catch {
        pricePart = '';
      }

      const ctx = Number.isFinite(m.context_length) ? Number(m.context_length) : (Number.isFinite(m.context) ? Number(m.context) : undefined);
      const ctxLabel = ctx ? ` - context ${Math.floor(ctx / 1000)}k` : '';

      const maxTokenAllowed = ctx ?? 8000;
      const maxCompletionTokens = Math.min(8192, maxTokenAllowed || 8192);

      const label = `${display}${pricePart}${ctxLabel}`;

      return {
        name: id,
        label,
        provider: this.name,
        maxTokenAllowed,
        maxCompletionTokens,
        // keep raw model for debugging if needed
        raw: m as any,
      } as unknown as ModelInfo;
    });

    // dedupe by name
    const unique = new Map<string, ModelInfo>();
    for (const mi of mapped) {
      if (!mi || !mi.name) continue;
      if (!unique.has(mi.name)) unique.set(mi.name, mi);
    }

    const dynamic = Array.from(unique.values());

    // If nothing discovered, return static models (fallback)
    return dynamic.length ? dynamic : this.staticModels || [];
  }

  /**
   * Create a model instance. This method is tolerant to multiple ways credentials can be provided.
   */
  getModelInstance(options: {
    model: string;
    serverEnv: ServerEnv;
    apiKeys?: Record<string, string>;
    providerSettings?: Record<string, IProviderSetting>;
  }): LanguageModelV1 {
    const { model, serverEnv = {}, apiKeys = {}, providerSettings = {} } = options;

    // Try existing BaseProvider helper first, but fall back to multiple candidates
    const tryBaseProvider = () => {
      try {
        return this.getProviderBaseUrlAndKey({
          apiKeys,
          providerSettings: providerSettings?.[this.name],
          serverEnv: serverEnv as any,
          defaultBaseUrlKey: 'ANTHROPIC_BASE_URL',
          defaultApiTokenKey: 'ANTHROPIC_API_KEY',
        });
      } catch {
        return { baseUrl: '', apiKey: '' };
      }
    };

    const primary = tryBaseProvider();
    let baseUrl = primary.baseUrl || '';
    let apiKey = primary.apiKey || '';

    // Fallbacks
    if (!apiKey) {
      apiKey =
        apiKeys[this.name] ||
        apiKeys.ANTHROPIC_API_KEY ||
        apiKeys.AGENTROUTER_API_KEY ||
        serverEnv.ANTHROPIC_API_KEY ||
        serverEnv.AGENTROUTER_API_KEY ||
        (providerSettings?.[this.name] as any)?.apiKey ||
        (providerSettings?.['Anthropic'] as any)?.apiKey ||
        '';
    }

    if (!baseUrl) {
      baseUrl =
        primary.baseUrl ||
        (providerSettings?.[this.name] as any)?.baseUrl ||
        (providerSettings?.[this.name] as any)?.base_url ||
        serverEnv.ANTHROPIC_BASE_URL ||
        serverEnv.AGENTROUTER_BASE_URL ||
        process.env?.ANTHROPIC_BASE_URL ||
        'https://agentrouter.org/';
    }

    if (!baseUrl || !apiKey) {
      // Provide detailed guidance in the error to help debug
      throw new Error(
        `Missing configuration for ${this.name} provider. Unable to find a baseUrl and/or apiKey. Tried:
- providerSettings['${this.name}'], providerSettings['Anthropic']
- apiKeys['${this.name}'], apiKeys['ANTHROPIC_API_KEY'], apiKeys['AGENTROUTER_API_KEY']
- serverEnv.ANTHROPIC_API_KEY, serverEnv.AGENTROUTER_API_KEY, serverEnv.ANTHROPIC_BASE_URL
Please pass credentials in one of those places. Example:
providerSettings = {
  "${this.name}": { apiKey: "<KEY>", baseUrl: "https://agentrouter.org/" }
}
Or set environment variables ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL.`
      );
    }

    return getOpenAILikeModel(baseUrl, apiKey, model);
  }
}
