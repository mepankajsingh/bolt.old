import { BaseProvider, getOpenAILikeModel } from '~/lib/modules/llm/base-provider';
import type { ModelInfo } from '~/lib/modules/llm/types';
import type { IProviderSetting } from '~/types/model';
import type { LanguageModelV1 } from 'ai';

export default class TogetherProvider extends BaseProvider {
  name = 'AgentRouter';
  getApiKeyLink = 'https://agentrouter.org/';

  config = {
    baseUrlKey: 'ANTHROPIC_BASE_URL',
    apiTokenKey: 'ANTHROPIC_API_KEY',
  };

  // made provider consistent with this.name
  staticModels: ModelInfo[] = [
    {
      name: 'claude-sonnet-4-5-20250929',
      label: 'Claude Sonnet 4.5',
      provider: this.name,
      maxTokenAllowed: 200000,
      maxCompletionTokens: 8192,
    }
  ];

  async getDynamicModels(
    apiKeys?: Record<string, string>,
    settings?: IProviderSetting,
    serverEnv: Record<string, string> = {},
  ): Promise<ModelInfo[]> {
    const { baseUrl: fetchBaseUrl, apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: settings,
      serverEnv,
      defaultBaseUrlKey: 'ANTHROPIC_BASE_URL',
      defaultApiTokenKey: 'ANTHROPIC_API_KEY',
    });
    const baseUrl = fetchBaseUrl || 'https://agentrouter.org/';

    if (!baseUrl || !apiKey) {
      return [];
    }

    let resJson: any;
    try {
      const response = await fetch(`${baseUrl.replace(/\/$/, '')}/v1beta/models`, {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          Accept: 'application/json',
        },
      });

      if (!response.ok) {
        // if non-2xx, try to parse body for debug but return []
        try {
          resJson = await response.json();
          // optional: console.warn('AgentRouter models fetch non-OK:', response.status, resJson);
        } catch {
          // console.warn('AgentRouter models fetch non-OK and body not json', response.statusText);
        }
        return [];
      }

      resJson = await response.json();
    } catch (err) {
      // network / parse error
      // console.error('Failed to fetch AgentRouter models', err);
      return [];
    }

    // AgentRouter may return: an array, or { models: [...] } or { data: [...] }
    let rawModels: any[] = [];
    if (Array.isArray(resJson)) {
      rawModels = resJson;
    } else if (Array.isArray(resJson.models)) {
      rawModels = resJson.models;
    } else if (Array.isArray(resJson.data)) {
      rawModels = resJson.data;
    } else {
      // unknown shape
      rawModels = [];
    }

    // filter only chat-like models, but be defensive if `type` missing
    const chatModels = rawModels.filter((m: any) => {
      if (!m) return false;
      if (typeof m.type === 'string') return m.type === 'chat';
      // sometimes model ids or names imply chat
      const idOrName = `${m.id || m.name || ''}`.toLowerCase();
      return idOrName.includes('chat') || idOrName.includes('claude') || idOrName.includes('sonnet');
    });

    // map & guard fields
    const mapped = chatModels.map((m: any) => {
      const id = m.id || m.model_id || m.name;
      const displayName = m.display_name || m.name || id || 'unknown';
      // pricing may be missing or have different shapes; guard it
      let priceLabel = '';
      try {
        if (m.pricing && typeof m.pricing.input === 'number' && typeof m.pricing.output === 'number') {
          priceLabel = ` - in:$${m.pricing.input.toFixed(2)} out:$${m.pricing.output.toFixed(2)}`;
        } else if (m.price && typeof m.price === 'number') {
          priceLabel = ` - $${m.price.toFixed(4)}`;
        }
      } catch {
        priceLabel = '';
      }

      const ctx = Number.isFinite(m.context_length) ? Number(m.context_length) : undefined;
      const ctxLabel = ctx ? ` - context ${Math.floor(ctx / 1000)}k` : '';

      const maxTokenAllowed = ctx ?? 8000;

      return {
        name: id,
        label: `${displayName}${priceLabel}${ctxLabel}`,
        provider: this.name,
        maxTokenAllowed,
        maxCompletionTokens: Math.min(8192, maxTokenAllowed || 8192),
        raw: m, // optional: keep raw object for debugging if needed
      } as ModelInfo;
    });

    // dedupe by name/id
    const unique = new Map<string, ModelInfo>();
    for (const mi of mapped) {
      if (!mi.name) continue;
      if (!unique.has(mi.name)) unique.set(mi.name, mi);
    }

    return Array.from(unique.values());
  }

  getModelInstance(options: {
    model: string;
    serverEnv: Env;
    apiKeys?: Record<string, string>;
    providerSettings?: Record<string, IProviderSetting>;
  }): LanguageModelV1 {
    const { model, serverEnv, apiKeys, providerSettings } = options;

    const { baseUrl, apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: providerSettings?.[this.name],
      serverEnv: serverEnv as any,
      defaultBaseUrlKey: 'ANTHROPIC_BASE_URL',
      defaultApiTokenKey: 'ANTHROPIC_API_KEY',
    });

    if (!baseUrl || !apiKey) {
      throw new Error(`Missing configuration for ${this.name} provider`);
    }

    return getOpenAILikeModel(baseUrl, apiKey, model);
  }
}
