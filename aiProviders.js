/**
 * AI Providers Module - Handle real AI API calls with different providers
 * Supports OpenRouter, HuggingFace, and Local (Ollama) endpoints
 */

// CORS proxy helper for endpoints that don't support CORS
const withCors = (url) => {
    return `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`;
};

/**
 * Send chat request to selected AI provider
 * @param {string} provider - Provider type: 'openrouter', 'huggingface', or 'local'
 * @param {string} model - Model ID to use
 * @param {Array} messages - Array of message objects with role and content
 * @param {string} apiKey - API key (optional for some providers)
 * @param {string} localEndpoint - Local endpoint URL (for local provider only)
 * @returns {Promise<string>} - Assistant response text
 */
async function sendChat(provider, model, messages, apiKey = '', localEndpoint = '') {
    try {
        switch (provider) {
            case 'openrouter':
                return await sendOpenRouterChat(model, messages, apiKey);
            
            case 'huggingface':
                return await sendHuggingFaceChat(model, messages, apiKey);
            
            case 'local':
                return await sendLocalChat(model, messages, localEndpoint);
            
            default:
                throw new Error(`Unsupported provider: ${provider}`);
        }
    } catch (error) {
        console.error('AI Provider Error:', error);
        throw new Error(`AI request failed: ${error.message}`);
    }
}

/**
 * OpenRouter API integration
 */
async function sendOpenRouterChat(model, messages, apiKey) {
    if (!apiKey) {
        throw new Error('OpenRouter API key is required');
    }

    const url = 'https://openrouter.ai/api/v1/chat/completions';
    const payload = {
        model: model,
        messages: messages,
        temperature: 0.7,
        max_tokens: 500
    };

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`,
                'HTTP-Referer': window.location.origin,
                'X-Title': 'AgentForge Lite'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error?.message || `HTTP ${response.status}`);
        }

        const data = await response.json();
        if (!data.choices || !data.choices[0]) {
            throw new Error('Invalid response format from OpenRouter');
        }

        return data.choices[0].message.content;
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('CORS')) {
            // Try with CORS proxy
            try {
                const corsResponse = await fetch(withCors(url), {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${apiKey}`,
                        'HTTP-Referer': window.location.origin,
                        'X-Title': 'AgentForge Lite'
                    },
                    body: JSON.stringify(payload)
                });

                if (!corsResponse.ok) {
                    throw new Error(`HTTP ${corsResponse.status}`);
                }

                const corsData = await corsResponse.json();
                return corsData.choices[0].message.content;
            } catch (corsError) {
                throw new Error('CORS error - try enabling browser CORS extension or use different provider');
            }
        }
        throw error;
    }
}

/**
 * HuggingFace Inference API integration
 */
async function sendHuggingFaceChat(model, messages, apiKey) {
    const url = `https://api-inference.huggingface.co/models/${model}`;
    
    // Convert messages to simple prompt format
    const prompt = messages.map(msg => {
        if (msg.role === 'system') return `System: ${msg.content}`;
        if (msg.role === 'user') return `User: ${msg.content}`;
        if (msg.role === 'assistant') return `Assistant: ${msg.content}`;
        return msg.content;
    }).join('\n') + '\nAssistant:';

    const payload = {
        inputs: prompt,
        parameters: {
            max_new_tokens: 500,
            temperature: 0.7,
            return_full_text: false
        }
    };

    const headers = {
        'Content-Type': 'application/json'
    };

    // Add API key if provided
    if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
    }

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error?.message || `HTTP ${response.status} - Model may be loading, try again in a moment`);
        }

        const data = await response.json();
        
        if (Array.isArray(data) && data[0]?.generated_text) {
            return data[0].generated_text.trim();
        } else if (data.generated_text) {
            return data.generated_text.trim();
        } else {
            throw new Error('Invalid response format from HuggingFace');
        }
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('CORS')) {
            // Try with CORS proxy
            try {
                const corsResponse = await fetch(withCors(url), {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify(payload)
                });

                if (!corsResponse.ok) {
                    throw new Error(`HTTP ${corsResponse.status}`);
                }

                const corsData = await corsResponse.json();
                return corsData[0]?.generated_text?.trim() || corsData.generated_text?.trim() || 'No response generated';
            } catch (corsError) {
                throw new Error('CORS error - HuggingFace may require API key or try different model');
            }
        }
        throw error;
    }
}

/**
 * Local Ollama API integration
 */
async function sendLocalChat(model, messages, endpoint) {
    if (!endpoint) {
        endpoint = 'http://localhost:11434/api/chat';
    }

    const payload = {
        model: model,
        messages: messages,
        stream: false,
        options: {
            temperature: 0.7,
            num_predict: 500
        }
    };

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Ollama server error: HTTP ${response.status}. Make sure Ollama is running and model '${model}' is installed.`);
        }

        const data = await response.json();
        
        if (data.message && data.message.content) {
            return data.message.content.trim();
        } else {
            throw new Error('Invalid response format from Ollama');
        }
    } catch (error) {
        if (error.name === 'TypeError' && (error.message.includes('Failed to fetch') || error.message.includes('CORS'))) {
            throw new Error('Cannot connect to Ollama server. Make sure Ollama is running on the specified endpoint and CORS is enabled.');
        }
        throw error;
    }
}

/**
 * Get default models for each provider
 */
function getDefaultModels() {
    return {
        openrouter: 'deepseek/deepseek-r1:free',
        huggingface: 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        local: 'llama3'
    };
}

/**
 * Get default endpoints
 */
function getDefaultEndpoints() {
    return {
        local: 'http://localhost:11434/api/chat'
    };
}

// Export functions for use in main application
window.aiProviders = {
    sendChat,
    getDefaultModels,
    getDefaultEndpoints,
    withCors
};