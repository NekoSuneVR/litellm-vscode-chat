import * as vscode from "vscode";
import {
	CancellationToken,
	LanguageModelChatInformation,
	LanguageModelChatRequestMessage,
	LanguageModelChatProvider,
	LanguageModelResponsePart,
	Progress,
	ProvideLanguageModelChatResponseOptions,
} from "vscode";

import type { HFModelItem, HFModelsResponse } from "./types";

import { convertTools, convertMessages, tryParseJSONObject, validateRequest } from "./utils";

const DEFAULT_MAX_OUTPUT_TOKENS = 16000;
const DEFAULT_CONTEXT_LENGTH = 128000;

/**
 * VS Code Chat provider backed by LiteLLM.
 */
export class LiteLLMChatModelProvider implements LanguageModelChatProvider {
	private _chatEndpoints: { model: string; modelMaxPromptTokens: number }[] = [];
	private _toolCallBuffers: Map<number, { id?: string; name?: string; args: string }> = new Map();
	private _completedToolCallIndices = new Set<number>();
	private _hasEmittedAssistantText = false;
	private _emittedBeginToolCallsHint = false;

	private _textToolParserBuffer = "";
	private _textToolActive:
		| undefined
		| {
			name?: string;
			index?: number;
			argBuffer: string;
			emitted?: boolean;
		};
	private _emittedTextToolCallKeys = new Set<string>();
	private _emittedTextToolCallIds = new Set<string>();

	constructor(private readonly secrets: vscode.SecretStorage, private readonly userAgent: string) { }

	private estimateMessagesTokens(msgs: readonly vscode.LanguageModelChatRequestMessage[]): number {
		let total = 0;
		for (const m of msgs) {
			for (const part of m.content) {
				if (part instanceof vscode.LanguageModelTextPart) {
					total += Math.ceil(part.value.length / 4);
				}
			}
		}
		return total;
	}

	private estimateToolTokens(tools: { type: string; function: { name: string; description?: string; parameters?: object } }[] | undefined): number {
		if (!tools || tools.length === 0) { return 0; }
		try {
			return Math.ceil(JSON.stringify(tools).length / 4);
		} catch {
			return 0;
		}
	}

	async prepareLanguageModelChatInformation(
		options: { silent: boolean },
		_token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {

		const config = await this.ensureConfig(options.silent);
		if (!config) return [];

		const { models } = await this.fetchModels(config.apiKey, config.baseUrl);

		const infos: LanguageModelChatInformation[] = models.map((m) => ({
			id: m.id,
			name: m.id,
			tooltip: "LiteLLM",
			family: "litellm",
			version: "1.0.0",
			maxInputTokens: Math.max(1, DEFAULT_CONTEXT_LENGTH - DEFAULT_MAX_OUTPUT_TOKENS),
			maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS,
			capabilities: {
				toolCalling: true,
				imageInput: false,
			},
		}));

		this._chatEndpoints = infos.map(i => ({
			model: i.id,
			modelMaxPromptTokens: i.maxInputTokens + i.maxOutputTokens,
		}));

		return infos;
	}

	async provideLanguageModelChatInformation(
		options: { silent: boolean },
		token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {
		return this.prepareLanguageModelChatInformation(options, token);
	}

	private async fetchModels(apiKey: string, baseUrl: string): Promise<{ models: HFModelItem[] }> {
		const headers: Record<string, string> = { "User-Agent": this.userAgent };
		if (apiKey) {
			headers.Authorization = `Bearer ${apiKey}`;
			headers["X-API-Key"] = apiKey;
		}

		const resp = await fetch(`${baseUrl}/v1/models`, { headers });
		if (!resp.ok) throw new Error(await resp.text());

		const parsed = (await resp.json()) as HFModelsResponse;
		return { models: parsed.data ?? [] };
	}

	async provideLanguageModelChatResponse(
		model: LanguageModelChatInformation,
		messages: readonly LanguageModelChatRequestMessage[],
		options: ProvideLanguageModelChatResponseOptions,
		progress: Progress<LanguageModelResponsePart>,
		token: CancellationToken
	): Promise<void> {

		this._toolCallBuffers.clear();
		this._completedToolCallIndices.clear();
		this._hasEmittedAssistantText = false;
		this._emittedBeginToolCallsHint = false;
		this._textToolParserBuffer = "";
		this._textToolActive = undefined;
		this._emittedTextToolCallKeys.clear();
		this._emittedTextToolCallIds.clear();

		const config = await this.ensureConfig(true);
		if (!config) throw new Error("LiteLLM not configured");

		validateRequest(messages);

		const requestBody: Record<string, unknown> = {
			model: model.id,
			messages: convertMessages(messages),
			stream: true,
			max_tokens: Math.min(options.modelOptions?.max_tokens || 4096, model.maxOutputTokens),
			temperature: options.modelOptions?.temperature ?? 0.7,
		};

		const toolConfig = convertTools(options);
		if (toolConfig.tools) requestBody.tools = toolConfig.tools;
		if (toolConfig.tool_choice) requestBody.tool_choice = toolConfig.tool_choice;

		const headers: Record<string, string> = {
			"Content-Type": "application/json",
			"User-Agent": this.userAgent,
		};
		if (config.apiKey) {
			headers.Authorization = `Bearer ${config.apiKey}`;
			headers["X-API-Key"] = config.apiKey;
		}

		const response = await fetch(`${config.baseUrl}/v1/chat/completions`, {
			method: "POST",
			headers,
			body: JSON.stringify(requestBody),
		});

		if (!response.ok) {
			throw new Error(await response.text());
		}

		// 🔧 FIX: handle non-streaming LiteLLM responses
		const contentType = response.headers.get("content-type") ?? "";
		if (!contentType.includes("text/event-stream")) {
			const json = await response.json();
			const text =
				json?.choices?.[0]?.message?.content ??
				json?.assistant?.response?.[0]?.content ??
				JSON.stringify(json, null, 2);

			progress.report(new vscode.LanguageModelTextPart(String(text)));
			return;
		}

		if (!response.body) {
			throw new Error("No response body");
		}

		await this.processStreamingResponse(response.body, progress, token);
	}

	async provideTokenCount(
		_model: LanguageModelChatInformation,
		text: string | LanguageModelChatRequestMessage
	): Promise<number> {
		if (typeof text === "string") return Math.ceil(text.length / 4);
		let total = 0;
		for (const part of text.content) {
			if (part instanceof vscode.LanguageModelTextPart) {
				total += Math.ceil(part.value.length / 4);
			}
		}
		return total;
	}

	private async ensureConfig(silent: boolean): Promise<{ baseUrl: string; apiKey: string } | undefined> {
		let baseUrl = await this.secrets.get("litellm.baseUrl");
		let apiKey = await this.secrets.get("litellm.apiKey");

		if (!baseUrl && !silent) {
			baseUrl = await vscode.window.showInputBox({ prompt: "LiteLLM Base URL" });
			if (baseUrl) await this.secrets.store("litellm.baseUrl", baseUrl);
		}

		if (!apiKey && !silent) {
			apiKey = await vscode.window.showInputBox({ prompt: "LiteLLM API Key (optional)" });
			if (apiKey) await this.secrets.store("litellm.apiKey", apiKey);
		}

		if (!baseUrl) return undefined;
		return { baseUrl, apiKey: apiKey ?? "" };
	}

	private async processStreamingResponse(
		responseBody: ReadableStream<Uint8Array>,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>,
		token: vscode.CancellationToken,
	): Promise<void> {

		const reader = responseBody.getReader();
		const decoder = new TextDecoder();
		let buffer = "";

		while (!token.isCancellationRequested) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split("\n");
			buffer = lines.pop() || "";

			for (const line of lines) {
				if (!line.startsWith("data: ")) continue;
				const data = line.slice(6);
				if (data === "[DONE]") return;

				try {
					const parsed = JSON.parse(data);
					const content = parsed?.choices?.[0]?.delta?.content;
					if (content) {
						progress.report(new vscode.LanguageModelTextPart(String(content)));
					}
				} catch { }
			}
		}
	}
}
