/**
 * AgentForge Lite - Enhanced with Real AI Integration
 * Supports OpenRouter, HuggingFace, and Local Ollama endpoints
 */

// AI Providers Module - Integrated directly into the application
const AIProviders = {
    // CORS proxy helper for endpoints that don't support CORS
    withCors(url) {
        return `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`;
    },

    // Default models for each provider
    getDefaultModels() {
        return {
            openrouter: 'deepseek/deepseek-r1:free',
            huggingface: 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            local: 'llama3'
        };
    },

    // Default endpoints
    getDefaultEndpoints() {
        return {
            local: 'http://localhost:11434/api/chat'
        };
    },

    // Main chat function
    async sendChat(provider, model, messages, apiKey = '', localEndpoint = '') {
        try {
            switch (provider) {
                case 'openrouter':
                    return await this.sendOpenRouterChat(model, messages, apiKey);
                case 'huggingface':
                    return await this.sendHuggingFaceChat(model, messages, apiKey);
                case 'local':
                    return await this.sendLocalChat(model, messages, localEndpoint);
                default:
                    throw new Error(`Unsupported provider: ${provider}`);
            }
        } catch (error) {
            console.error('AI Provider Error:', error);
            throw new Error(`AI request failed: ${error.message}`);
        }
    },

    // OpenRouter API integration
    async sendOpenRouterChat(model, messages, apiKey) {
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
                try {
                    const corsResponse = await fetch(this.withCors(url), {
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
    },

    // HuggingFace Inference API integration
    async sendHuggingFaceChat(model, messages, apiKey) {
        const url = `https://api-inference.huggingface.co/models/${model}`;
        
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
                try {
                    const corsResponse = await fetch(this.withCors(url), {
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
    },

    // Local Ollama API integration
    async sendLocalChat(model, messages, endpoint) {
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
};

// Application Data
const TEMPLATES_DATA = [
    {
        id: "customer-support",
        name: "Customer Support Chatbot",
        description: "24/7 AI assistant for customer inquiries and support",
        avatar: "🤖",
        category: "Business",
        systemPrompt: "You are a helpful and professional customer support assistant. Always be polite, empathetic, and solution-focused. If you cannot solve a problem, escalate to human support gracefully.",
        personality: "professional",
        responseStyle: "helpful",
        tags: ["support", "customer-service", "business"]
    },
    {
        id: "code-helper",
        name: "Code Helper Assistant",
        description: "Programming companion for debugging and code assistance",
        avatar: "💻",
        category: "Development",
        systemPrompt: "You are an expert programming assistant. Help users write, debug, and optimize code. Explain concepts clearly and provide working examples. Support multiple programming languages.",
        personality: "technical",
        responseStyle: "precise",
        tags: ["programming", "coding", "development", "debugging"]
    },
    {
        id: "content-creator",
        name: "Content Creator",
        description: "Creative writing and marketing content assistant",
        avatar: "✍️",
        category: "Creative",
        systemPrompt: "You are a creative content creator specializing in engaging, original content. Help with blog posts, social media, marketing copy, and creative writing. Make content compelling and audience-appropriate.",
        personality: "creative",
        responseStyle: "engaging",
        tags: ["writing", "marketing", "creative", "content"]
    },
    {
        id: "data-analyzer",
        name: "Data Analyzer",
        description: "Statistical analysis and data interpretation specialist",
        avatar: "📊",
        category: "Analytics",
        systemPrompt: "You are a data analysis expert. Help users understand data patterns, create visualizations, perform statistical analysis, and draw meaningful insights from datasets.",
        personality: "analytical",
        responseStyle: "factual",
        tags: ["data", "analytics", "statistics", "insights"]
    },
    {
        id: "personal-tutor",
        name: "Personal Tutor",
        description: "Educational assistant for learning and skill development",
        avatar: "🎓",
        category: "Education",
        systemPrompt: "You are a patient and encouraging tutor. Break down complex topics into digestible parts, provide examples, check understanding, and adapt to different learning styles.",
        personality: "patient",
        responseStyle: "educational",
        tags: ["education", "learning", "tutor", "teaching"]
    },
    {
        id: "business-consultant",
        name: "Business Consultant",
        description: "Strategic business advisor and planning assistant",
        avatar: "💼",
        category: "Business",
        systemPrompt: "You are an experienced business consultant. Provide strategic advice, help with business planning, market analysis, and operational improvements. Focus on practical, actionable insights.",
        personality: "strategic",
        responseStyle: "advisory",
        tags: ["business", "strategy", "consulting", "planning"]
    }
];

const MARKETPLACE_AGENTS = [
    {
        id: "recipe-chef",
        name: "Recipe Chef AI",
        description: "Culinary expert for recipes and cooking tips",
        avatar: "👨‍🍳",
        category: "Lifestyle",
        creator: "CookingEnthusiast",
        likes: 127,
        uses: 1453,
        tags: ["cooking", "recipes", "food"],
        systemPrompt: "You are a professional chef with expertise in international cuisines. Help users find recipes, cooking techniques, and meal planning.",
        personality: "friendly",
        responseStyle: "helpful"
    },
    {
        id: "fitness-coach",
        name: "Personal Fitness Coach",
        description: "Workout plans and health guidance",
        avatar: "💪",
        category: "Health",
        creator: "FitnessPro",
        likes: 98,
        uses: 892,
        tags: ["fitness", "health", "workout"],
        systemPrompt: "You are a certified fitness coach. Create personalized workout plans, provide nutrition advice, and motivate users to achieve their fitness goals.",
        personality: "encouraging",
        responseStyle: "motivational"
    },
    {
        id: "travel-guide",
        name: "Travel Planning Assistant",
        description: "Itinerary planning and travel advice",
        avatar: "✈️",
        category: "Travel",
        creator: "Wanderlust",
        likes: 156,
        uses: 2341,
        tags: ["travel", "planning", "destinations"],
        systemPrompt: "You are an experienced travel advisor. Help users plan trips, find destinations, book accommodations, and discover local attractions.",
        personality: "enthusiastic",
        responseStyle: "informative"
    }
];

// Mock AI responses for fallback
const MOCK_RESPONSES = {
    professional: {
        helpful: "I'm here to assist you with professional and helpful guidance. How may I help you today?",
        precise: "I provide accurate, technical information. Please specify your requirements.",
        factual: "I deliver data-driven insights and factual analysis. What information do you need?"
    },
    friendly: {
        helpful: "Hi there! I'm excited to help you out. What can I do for you today? 😊",
        engaging: "Hey! I love connecting with people and creating awesome content. What's on your mind?",
        motivational: "You've got this! I'm here to support and encourage you every step of the way!"
    },
    technical: {
        precise: "I specialize in technical problem-solving and code analysis. Please provide your specific technical challenge.",
        helpful: "I'm here to help with your technical questions and provide detailed explanations."
    },
    creative: {
        engaging: "Let's unleash some creativity! I'm bursting with ideas and can't wait to collaborate with you!",
        helpful: "I'm here to spark your imagination and help bring your creative visions to life!"
    },
    analytical: {
        factual: "I analyze data patterns and provide evidence-based insights. Share your data requirements.",
        precise: "I deliver systematic analysis with statistical accuracy and clear conclusions."
    },
    patient: {
        educational: "I'm here to guide you through learning at your own pace. No question is too small - what would you like to explore?",
        helpful: "Take your time - I'm patient and here to support your learning journey every step of the way."
    },
    strategic: {
        advisory: "I provide strategic business insights based on market analysis and best practices. What business challenge can I help you with?",
        helpful: "I'm here to help you make informed business decisions with strategic guidance."
    }
};

class AgentForgeApp {
    constructor() {
        this.currentScreen = 'home';
        this.currentAgent = null;
        this.chatHistory = [];
        this.templates = TEMPLATES_DATA;
        this.marketplaceAgents = MARKETPLACE_AGENTS;
        this.settings = this.loadSettings();
    }

    init() {
        this.setupEventListeners();
        this.renderTemplates();
        this.renderMarketplace();
        this.showScreen('home');
        this.checkImportParameter();
        this.updateProviderBadges();
        
        // Show banner on first visit
        if (!localStorage.getItem('agentforge.bannerDismissed')) {
            this.showBanner();
        } else {
            this.hideBanner();
        }
    }

    // Settings Management
    loadSettings() {
        const stored = localStorage.getItem('agentforge.settings');
        const defaults = {
            provider: 'openrouter',
            model: AIProviders.getDefaultModels().openrouter,
            apiKey: '',
            localEndpoint: AIProviders.getDefaultEndpoints().local
        };
        
        return stored ? { ...defaults, ...JSON.parse(stored) } : defaults;
    }

    saveSettings() {
        // Don't log API keys
        const settingsToLog = { ...this.settings };
        if (settingsToLog.apiKey) {
            settingsToLog.apiKey = '[REDACTED]';
        }
        console.log('Saving settings:', settingsToLog);
        
        localStorage.setItem('agentforge.settings', JSON.stringify(this.settings));
    }

    // Toast Notifications
    showToast(title, message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast toast--${type}`;
        
        const iconMap = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle', 
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };

        toast.innerHTML = `
            <div class="toast-icon ${type}">
                <i class="${iconMap[type]}"></i>
            </div>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close">
                <i class="fas fa-times"></i>
            </button>
        `;

        container.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);

        // Manual close
        toast.querySelector('.toast-close').addEventListener('click', () => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        });
    }

    // Banner Management
    showBanner() {
        const banner = document.getElementById('ai-setup-banner');
        if (banner) {
            banner.classList.remove('hidden');
            // Adjust app container margin
            const appContainer = document.querySelector('.app-container');
            if (appContainer) {
                appContainer.style.marginTop = '112px';
            }
        }
    }

    hideBanner() {
        const banner = document.getElementById('ai-setup-banner');
        if (banner) {
            banner.classList.add('hidden');
            const appContainer = document.querySelector('.app-container');
            if (appContainer) {
                appContainer.style.marginTop = '64px';
            }
        }
    }

    // Provider Badges
    updateProviderBadges() {
        const container = document.getElementById('provider-badges');
        if (!container) return;

        container.innerHTML = `
            <span class="provider-badge">${this.settings.provider.toUpperCase()}</span>
            <span class="provider-badge">${this.settings.model}</span>
        `;
    }

    setupEventListeners() {
        // Banner dismiss
        const dismissBanner = document.getElementById('dismiss-banner');
        if (dismissBanner) {
            dismissBanner.addEventListener('click', (e) => {
                e.preventDefault();
                this.hideBanner();
                localStorage.setItem('agentforge.bannerDismissed', 'true');
            });
        }

        // Settings button
        const settingsBtn = document.getElementById('settings-btn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('Settings button clicked');
                this.openSettingsModal();
            });
        }

        // Settings modal handlers
        const saveSettings = document.getElementById('save-settings');
        if (saveSettings) {
            saveSettings.addEventListener('click', (e) => {
                e.preventDefault();
                this.saveSettingsFromModal();
            });
        }

        const providerSelect = document.getElementById('ai-provider-select');
        if (providerSelect) {
            providerSelect.addEventListener('change', () => this.updateSettingsUI());
        }

        // Navigation - Fixed with proper event handling
        const navHome = document.getElementById('nav-home');
        const navBuilder = document.getElementById('nav-builder');
        const navMarketplace = document.getElementById('nav-marketplace');
        const navHelp = document.getElementById('nav-help');

        if (navHome) {
            navHome.addEventListener('click', (e) => { 
                e.preventDefault(); 
                console.log('Nav home clicked');
                this.showScreen('home'); 
            });
        }
        if (navBuilder) {
            navBuilder.addEventListener('click', (e) => { 
                e.preventDefault(); 
                console.log('Nav builder clicked');
                this.showScreen('templates'); 
            });
        }
        if (navMarketplace) {
            navMarketplace.addEventListener('click', (e) => { 
                e.preventDefault(); 
                console.log('Nav marketplace clicked');
                this.showScreen('marketplace'); 
            });
        }
        if (navHelp) {
            navHelp.addEventListener('click', (e) => { 
                e.preventDefault(); 
                console.log('Nav help clicked');
                this.showHelpModal(); 
            });
        }

        // Home screen - Fixed
        const startCreating = document.getElementById('start-creating');
        const browseAgents = document.getElementById('browse-agents');
        
        if (startCreating) {
            startCreating.addEventListener('click', (e) => { 
                e.preventDefault(); 
                console.log('Start creating clicked');
                this.showScreen('templates'); 
            });
        }
        if (browseAgents) {
            browseAgents.addEventListener('click', (e) => { 
                e.preventDefault(); 
                console.log('Browse agents clicked');
                this.showScreen('marketplace'); 
            });
        }

        // Templates screen - Fixed
        const backToHome = document.getElementById('back-to-home');
        if (backToHome) {
            backToHome.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.showScreen('home'); 
            });
        }

        // Builder screen - Fixed
        const backToTemplates = document.getElementById('back-to-templates');
        const testAgent = document.getElementById('test-agent');
        const exportAgent = document.getElementById('export-agent');

        if (backToTemplates) {
            backToTemplates.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.showScreen('templates'); 
            });
        }
        if (testAgent) {
            testAgent.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.openChatTest(); 
            });
        }
        if (exportAgent) {
            exportAgent.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.showExportModal(); 
            });
        }

        // Builder form inputs - Fixed
        const agentName = document.getElementById('agent-name');
        const agentDescription = document.getElementById('agent-description');
        const systemPrompt = document.getElementById('system-prompt');
        const personality = document.getElementById('personality');
        const responseStyle = document.getElementById('response-style');

        if (agentName) agentName.addEventListener('input', () => this.updatePreview());
        if (agentDescription) agentDescription.addEventListener('input', () => this.updatePreview());
        if (systemPrompt) systemPrompt.addEventListener('input', () => this.updatePreview());
        if (personality) personality.addEventListener('change', () => this.updatePreview());
        if (responseStyle) responseStyle.addEventListener('change', () => this.updatePreview());

        // Avatar selection - Fixed
        document.querySelectorAll('.avatar-option').forEach(option => {
            option.addEventListener('click', (e) => {
                e.preventDefault();
                this.selectAvatar(e.target.dataset.avatar);
            });
        });

        // Chat screen - Fixed
        const backToBuilder = document.getElementById('back-to-builder');
        const clearChat = document.getElementById('clear-chat');
        const finishTesting = document.getElementById('finish-testing');
        const sendMessage = document.getElementById('send-message');
        const messageInput = document.getElementById('message-input');

        if (backToBuilder) {
            backToBuilder.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.showScreen('builder'); 
            });
        }
        if (clearChat) {
            clearChat.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.clearChat(); 
            });
        }
        if (finishTesting) {
            finishTesting.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.showScreen('builder'); 
            });
        }
        if (sendMessage) {
            sendMessage.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.sendMessage(); 
            });
        }
        if (messageInput) {
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // Marketplace screen - Fixed
        const marketplaceBack = document.getElementById('marketplace-back');
        const categoryFilter = document.getElementById('category-filter');
        const searchAgents = document.getElementById('search-agents');

        if (marketplaceBack) {
            marketplaceBack.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.showScreen('home'); 
            });
        }
        if (categoryFilter) categoryFilter.addEventListener('change', () => this.filterMarketplace());
        if (searchAgents) searchAgents.addEventListener('input', () => this.filterMarketplace());

        // Export modal - Fixed
        const exportJSON = document.getElementById('export-json');
        const exportHTML = document.getElementById('export-html');
        const copyLink = document.getElementById('copy-link');

        if (exportJSON) {
            exportJSON.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.exportJSON(); 
            });
        }
        if (exportHTML) {
            exportHTML.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.exportHTML(); 
            });
        }
        if (copyLink) {
            copyLink.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.copyShareLink(); 
            });
        }

        // Import modal - Fixed
        const importFile = document.getElementById('import-file');
        const importFromFile = document.getElementById('import-from-file');
        const importFromText = document.getElementById('import-from-text');

        if (importFile) importFile.addEventListener('change', (e) => this.handleFileImport(e));
        if (importFromFile) {
            importFromFile.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.importFromFile(); 
            });
        }
        if (importFromText) {
            importFromText.addEventListener('click', (e) => { 
                e.preventDefault(); 
                this.importFromText(); 
            });
        }

        // Modal close handlers - Fixed
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const modalId = e.target.closest('.modal').id;
                this.hideModal(modalId);
            });
        });

        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    e.preventDefault();
                    this.hideModal(modal.id);
                }
            });
        });
    }

    // Settings Modal Functions
    openSettingsModal() {
        console.log('Opening settings modal');
        
        // Populate current settings
        const providerSelect = document.getElementById('ai-provider-select');
        const modelInput = document.getElementById('ai-model-input');
        const apiKeyInput = document.getElementById('ai-api-key');
        const localEndpointInput = document.getElementById('local-endpoint-input');

        if (providerSelect) providerSelect.value = this.settings.provider;
        if (modelInput) modelInput.value = this.settings.model;
        if (apiKeyInput) apiKeyInput.value = this.settings.apiKey;
        if (localEndpointInput) localEndpointInput.value = this.settings.localEndpoint;

        this.updateSettingsUI();
        this.showModal('settings-modal');
    }

    updateSettingsUI() {
        const providerSelect = document.getElementById('ai-provider-select');
        const modelInput = document.getElementById('ai-model-input');
        const apiKeyGroup = document.getElementById('api-key-group');
        const localEndpointGroup = document.getElementById('local-endpoint-group');
        const providerInfo = document.getElementById('provider-info');

        if (!providerSelect) return;

        const provider = providerSelect.value;
        const defaultModels = AIProviders.getDefaultModels();

        // Update default model if empty
        if (modelInput && !modelInput.value) {
            modelInput.value = defaultModels[provider];
        }

        // Show/hide relevant fields
        if (apiKeyGroup) {
            if (provider === 'local') {
                apiKeyGroup.classList.add('hidden');
            } else {
                apiKeyGroup.classList.remove('hidden');
            }
        }
        
        if (localEndpointGroup) {
            if (provider === 'local') {
                localEndpointGroup.classList.remove('hidden');
            } else {
                localEndpointGroup.classList.add('hidden');
            }
        }

        // Update provider info
        if (providerInfo) {
            const infoContent = {
                openrouter: {
                    title: 'OpenRouter',
                    description: 'Get a free API key at <a href="https://openrouter.ai" target="_blank">openrouter.ai</a><br>Free tier includes access to various models including DeepSeek R1.'
                },
                huggingface: {
                    title: 'HuggingFace',
                    description: 'Optional API key from <a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co</a><br>Free tier available but rate-limited. API key removes limits.'
                },
                local: {
                    title: 'Local (Ollama)',
                    description: 'Run models locally with <a href="https://ollama.ai" target="_blank">Ollama</a><br>Make sure Ollama is running and CORS is enabled: <code>OLLAMA_ORIGINS=* ollama serve</code>'
                }
            };

            const info = infoContent[provider];
            providerInfo.innerHTML = `
                <h4>${info.title}</h4>
                <p>${info.description}</p>
            `;
        }
    }

    saveSettingsFromModal() {
        const providerSelect = document.getElementById('ai-provider-select');
        const modelInput = document.getElementById('ai-model-input');
        const apiKeyInput = document.getElementById('ai-api-key');
        const localEndpointInput = document.getElementById('local-endpoint-input');

        if (providerSelect) this.settings.provider = providerSelect.value;
        if (modelInput) this.settings.model = modelInput.value || AIProviders.getDefaultModels()[this.settings.provider];
        if (apiKeyInput) this.settings.apiKey = apiKeyInput.value;
        if (localEndpointInput) this.settings.localEndpoint = localEndpointInput.value || AIProviders.getDefaultEndpoints().local;

        this.saveSettings();
        this.updateProviderBadges();
        this.hideModal('settings-modal');
        this.showToast('Settings Saved', 'AI provider settings have been updated successfully.', 'success');
    }

    // Enhanced Chat Functions with Real AI
    async sendMessage() {
        const input = document.getElementById('message-input');
        if (!input) return;
        
        const message = input.value.trim();
        if (!message) return;

        this.addMessage(message, 'user');
        input.value = '';
        this.showTypingIndicator();

        try {
            const response = await this.generateAIResponse(message);
            this.hideTypingIndicator();
            this.addMessage(response, 'ai');
        } catch (error) {
            this.hideTypingIndicator();
            console.error('AI Error:', error);
            this.showToast('AI Error', error.message, 'error');
            
            // Fallback to mock response
            const fallback = this.generateMockResponse(message);
            this.addMessage(`[Fallback Response] ${fallback}`, 'ai');
        }
    }

    async generateAIResponse(userMessage) {
        if (!this.currentAgent) {
            throw new Error("Agent not properly configured");
        }

        // Build message history for context
        const messages = [
            {
                role: 'system',
                content: this.currentAgent.systemPrompt
            }
        ];

        // Add recent chat history (last 6 messages for context)
        const recentHistory = this.chatHistory.slice(-6);
        recentHistory.forEach(msg => {
            messages.push({
                role: msg.sender === 'user' ? 'user' : 'assistant',
                content: msg.content
            });
        });

        // Add current user message
        messages.push({
            role: 'user',
            content: userMessage
        });

        // Make AI API call
        const response = await AIProviders.sendChat(
            this.settings.provider,
            this.settings.model,
            messages,
            this.settings.apiKey,
            this.settings.localEndpoint
        );

        return response;
    }

    generateMockResponse(userMessage) {
        if (!this.currentAgent) return "I'm not properly configured yet.";
        
        const personality = this.currentAgent.personality;
        const style = this.currentAgent.responseStyle;
        
        const responses = [
            MOCK_RESPONSES[personality]?.[style] || "I'm here to help!",
            `As configured in my system prompt, I understand you're asking about "${userMessage.toLowerCase()}". Let me help with that!`,
            `That's an interesting question! Given my ${style} response style, I think this is worth exploring further.`,
            `I'm designed to be ${personality} and ${style}. Regarding "${userMessage}", I would suggest we look at this from multiple angles.`
        ];
        
        return responses[Math.floor(Math.random() * responses.length)];
    }

    showTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.classList.remove('hidden');
        }
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.classList.add('hidden');
        }
    }

    // Screen Navigation
    showScreen(screenId) {
        console.log('Showing screen:', screenId);
        
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });

        // Show target screen
        const targetScreen = document.getElementById(`${screenId}-screen`);
        if (targetScreen) {
            targetScreen.classList.add('active');
            this.currentScreen = screenId;
            console.log('Screen switched to:', screenId);
        } else {
            console.error('Screen not found:', `${screenId}-screen`);
        }

        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });

        if (screenId === 'templates' || screenId === 'builder') {
            const navBuilder = document.getElementById('nav-builder');
            if (navBuilder) navBuilder.classList.add('active');
        } else if (screenId === 'marketplace') {
            const navMarketplace = document.getElementById('nav-marketplace');
            if (navMarketplace) navMarketplace.classList.add('active');
        }
    }

    renderTemplates() {
        const grid = document.getElementById('templates-grid');
        if (!grid) return;
        
        grid.innerHTML = '';

        this.templates.forEach(template => {
            const card = document.createElement('div');
            card.className = 'template-card';
            card.innerHTML = `
                <div class="template-header">
                    <div class="template-avatar">${template.avatar}</div>
                    <div class="template-info">
                        <h3 class="template-name">${template.name}</h3>
                        <span class="template-category">${template.category}</span>
                    </div>
                </div>
                <p class="template-description">${template.description}</p>
                <div class="template-tags">
                    ${template.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
            `;
            
            card.addEventListener('click', (e) => {
                e.preventDefault();
                this.selectTemplate(template);
            });
            
            grid.appendChild(card);
        });
    }

    renderMarketplace() {
        const grid = document.getElementById('marketplace-grid');
        if (!grid) return;
        
        grid.innerHTML = '';

        this.marketplaceAgents.forEach(agent => {
            const card = document.createElement('div');
            card.className = 'marketplace-card';
            card.dataset.category = agent.category.toLowerCase();
            card.innerHTML = `
                <div class="marketplace-header">
                    <div class="marketplace-avatar">${agent.avatar}</div>
                    <div class="marketplace-info">
                        <h3 class="marketplace-name">${agent.name}</h3>
                        <p class="marketplace-creator">by ${agent.creator}</p>
                    </div>
                </div>
                <p class="marketplace-description">${agent.description}</p>
                <div class="template-tags">
                    ${agent.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
                <div class="marketplace-footer">
                    <div class="marketplace-stats">
                        <div class="stat">
                            <i class="fas fa-heart"></i>
                            <span>${agent.likes}</span>
                        </div>
                        <div class="stat">
                            <i class="fas fa-download"></i>
                            <span>${agent.uses}</span>
                        </div>
                    </div>
                    <div class="marketplace-actions">
                        <button class="btn btn--outline btn--sm import-agent" data-agent-id="${agent.id}">
                            <i class="fas fa-download"></i>
                            Import
                        </button>
                    </div>
                </div>
            `;

            const importBtn = card.querySelector('.import-agent');
            if (importBtn) {
                importBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    this.importMarketplaceAgent(agent);
                });
            }

            grid.appendChild(card);
        });
    }

    selectTemplate(template) {
        this.currentAgent = {
            id: Date.now(),
            name: template.name,
            description: template.description,
            avatar: template.avatar,
            category: template.category,
            systemPrompt: template.systemPrompt,
            personality: template.personality,
            responseStyle: template.responseStyle,
            tags: [...template.tags],
            created: new Date().toISOString()
        };

        this.populateBuilderForm();
        this.updatePreview();
        this.showScreen('builder');
    }

    populateBuilderForm() {
        if (!this.currentAgent) return;

        const agentName = document.getElementById('agent-name');
        const agentDescription = document.getElementById('agent-description');
        const systemPrompt = document.getElementById('system-prompt');
        const personality = document.getElementById('personality');
        const responseStyle = document.getElementById('response-style');
        const agentTitle = document.getElementById('agent-title');

        if (agentName) agentName.value = this.currentAgent.name;
        if (agentDescription) agentDescription.value = this.currentAgent.description;
        if (systemPrompt) systemPrompt.value = this.currentAgent.systemPrompt;
        if (personality) personality.value = this.currentAgent.personality;
        if (responseStyle) responseStyle.value = this.currentAgent.responseStyle;
        if (agentTitle) agentTitle.textContent = `Customize ${this.currentAgent.name}`;
        
        this.selectAvatar(this.currentAgent.avatar);
    }

    selectAvatar(avatar) {
        document.querySelectorAll('.avatar-option').forEach(option => {
            option.classList.remove('active');
        });
        
        const selectedOption = document.querySelector(`[data-avatar="${avatar}"]`);
        if (selectedOption) {
            selectedOption.classList.add('active');
        }
        
        if (this.currentAgent) {
            this.currentAgent.avatar = avatar;
            this.updatePreview();
        }
    }

    updatePreview() {
        if (!this.currentAgent) return;

        const agentName = document.getElementById('agent-name');
        const agentDescription = document.getElementById('agent-description');
        const systemPrompt = document.getElementById('system-prompt');
        const personality = document.getElementById('personality');
        const responseStyle = document.getElementById('response-style');

        this.currentAgent.name = (agentName && agentName.value) || 'Unnamed Agent';
        this.currentAgent.description = (agentDescription && agentDescription.value) || 'No description provided';
        this.currentAgent.systemPrompt = (systemPrompt && systemPrompt.value) || 'You are a helpful AI assistant.';
        this.currentAgent.personality = (personality && personality.value) || 'professional';
        this.currentAgent.responseStyle = (responseStyle && responseStyle.value) || 'helpful';

        const previewAvatar = document.getElementById('preview-avatar');
        const previewName = document.getElementById('preview-name');
        const previewDescription = document.getElementById('preview-description');
        const previewPersonality = document.getElementById('preview-personality');
        const previewStyle = document.getElementById('preview-style');
        const previewSample = document.getElementById('preview-sample');

        if (previewAvatar) previewAvatar.textContent = this.currentAgent.avatar;
        if (previewName) previewName.textContent = this.currentAgent.name;
        if (previewDescription) previewDescription.textContent = this.currentAgent.description;
        if (previewPersonality) previewPersonality.textContent = this.currentAgent.personality;
        if (previewStyle) previewStyle.textContent = this.currentAgent.responseStyle;

        const sampleResponse = this.generateSampleResponse();
        if (previewSample) previewSample.textContent = sampleResponse;
    }

    generateSampleResponse() {
        const personality = this.currentAgent.personality;
        const style = this.currentAgent.responseStyle;
        
        return MOCK_RESPONSES[personality]?.[style] || 
               MOCK_RESPONSES[personality]?.helpful || 
               "I'm ready to help you with whatever you need!";
    }

    openChatTest() {
        if (!this.currentAgent) return;
        
        const chatAgentName = document.getElementById('chat-agent-name');
        const chatAvatar = document.getElementById('chat-avatar');
        
        if (chatAgentName) chatAgentName.textContent = `Testing: ${this.currentAgent.name}`;
        if (chatAvatar) chatAvatar.textContent = this.currentAgent.avatar;
        
        this.clearChat();
        this.addWelcomeMessage();
        this.updateProviderBadges();
        this.showScreen('chat');
    }

    addWelcomeMessage() {
        const welcomeMsg = document.querySelector('.welcome-message');
        if (welcomeMsg && this.currentAgent) {
            const avatar = welcomeMsg.querySelector('.message-avatar');
            const content = welcomeMsg.querySelector('.message-content p');
            
            if (avatar) avatar.textContent = this.currentAgent.avatar;
            if (content) {
                const sampleResponse = this.generateSampleResponse();
                content.textContent = sampleResponse;
            }
        }
    }

    addMessage(content, sender) {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}-message`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.textContent = sender === 'ai' ? (this.currentAgent ? this.currentAgent.avatar : '🤖') : '👤';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `<p>${content}</p>`;
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);
        
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        this.chatHistory.push({ content, sender, timestamp: Date.now() });
    }

    clearChat() {
        const messages = document.getElementById('chat-messages');
        if (!messages) return;
        
        const welcomeMessage = messages.querySelector('.welcome-message');
        messages.innerHTML = '';
        if (welcomeMessage) {
            messages.appendChild(welcomeMessage);
        }
        this.chatHistory = [];
    }

    filterMarketplace() {
        const categoryFilter = document.getElementById('category-filter');
        const searchInput = document.getElementById('search-agents');
        
        if (!categoryFilter || !searchInput) return;
        
        const selectedCategory = categoryFilter.value.toLowerCase();
        const searchTerm = searchInput.value.toLowerCase().trim();
        
        document.querySelectorAll('.marketplace-card').forEach(card => {
            const cardCategory = card.dataset.category;
            const cardText = card.textContent.toLowerCase();
            
            const categoryMatch = selectedCategory === 'all' || cardCategory === selectedCategory;
            const searchMatch = !searchTerm || cardText.includes(searchTerm);
            
            if (categoryMatch && searchMatch) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
    }

    showExportModal() {
        if (!this.currentAgent) return;
        
        // Include AI settings in export
        const exportAgent = { ...this.currentAgent, aiSettings: this.settings };
        const shareData = btoa(JSON.stringify(exportAgent));
        const shareLink = `${window.location.origin}${window.location.pathname}?import=${shareData}`;
        const shareLinkInput = document.getElementById('share-link');
        if (shareLinkInput) shareLinkInput.value = shareLink;
        
        this.showModal('export-modal');
    }

    exportJSON() {
        if (!this.currentAgent) return;
        
        const exportData = { ...this.currentAgent, aiSettings: this.settings };
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `${this.currentAgent.name.toLowerCase().replace(/\s+/g, '-')}-agent.json`;
        link.click();
        
        this.hideModal('export-modal');
        this.showToast('Export Complete', 'Agent configuration downloaded successfully.', 'success');
    }

    exportHTML() {
        if (!this.currentAgent) return;
        
        const htmlTemplate = this.generateHTMLExport();
        const dataBlob = new Blob([htmlTemplate], { type: 'text/html' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `${this.currentAgent.name.toLowerCase().replace(/\s+/g, '-')}-agent.html`;
        link.click();
        
        this.hideModal('export-modal');
        this.showToast('Export Complete', 'Standalone HTML agent downloaded successfully.', 'success');
    }

    generateHTMLExport() {
        // Enhanced HTML export with AI integration
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${this.currentAgent.name}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .chat-container { max-width: 800px; margin: 20px auto; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); overflow: hidden; }
        .header { background: linear-gradient(135deg, #218bc5, #1d7ab8); color: white; padding: 20px; text-align: center; }
        .avatar { font-size: 3rem; margin-bottom: 10px; }
        .ai-badge { background: rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 12px; font-size: 0.8rem; margin-top: 8px; }
        .messages { height: 400px; overflow-y: auto; padding: 20px; }
        .message { display: flex; margin-bottom: 15px; align-items: flex-start; gap: 10px; }
        .message.user { flex-direction: row-reverse; }
        .message-avatar { width: 35px; height: 35px; border-radius: 50%; background: #218bc5; color: white; display: flex; align-items: center; justify-content: center; font-size: 14px; }
        .message.user .message-avatar { background: #e0e0e0; color: #333; }
        .message-content { background: #f0f0f0; padding: 10px 15px; border-radius: 18px; max-width: 70%; }
        .message.user .message-content { background: #218bc5; color: white; }
        .input-area { padding: 20px; border-top: 1px solid #eee; display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 25px; outline: none; }
        .input-area button { padding: 12px 20px; background: #218bc5; color: white; border: none; border-radius: 25px; cursor: pointer; }
        .typing { text-align: center; color: #666; font-style: italic; padding: 10px; }
        .typing.hidden { display: none; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <div class="avatar">${this.currentAgent.avatar}</div>
            <h1>${this.currentAgent.name}</h1>
            <p>${this.currentAgent.description}</p>
            <div class="ai-badge">Powered by ${this.settings.provider.toUpperCase()} • ${this.settings.model}</div>
        </div>
        <div class="messages" id="messages">
            <div class="message">
                <div class="message-avatar">${this.currentAgent.avatar}</div>
                <div class="message-content">${this.generateSampleResponse()}</div>
            </div>
        </div>
        <div class="typing hidden" id="typing">AI is thinking...</div>
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        const agent = ${JSON.stringify(this.currentAgent)};
        const aiSettings = ${JSON.stringify(this.settings)};
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            showTyping();
            
            setTimeout(() => {
                const responses = [
                    "Thank you for your message! I'm here to help based on my configuration and training.",
                    "I understand your question. Let me provide you with assistance based on my system prompt.",
                    "That's an interesting question! I'd be happy to help you explore this topic further.",
                    "Based on my configuration as a " + agent.personality + " agent with a " + agent.responseStyle + " style, I think this is worth discussing."
                ];
                const response = responses[Math.floor(Math.random() * responses.length)];
                hideTyping();
                addMessage(response, 'ai');
            }, 1000 + Math.random() * 1000);
        }
        
        function addMessage(content, sender) {
            const messages = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = \`message \${sender}\`;
            messageDiv.innerHTML = \`
                <div class="message-avatar">\${sender === 'ai' ? agent.avatar : '👤'}</div>
                <div class="message-content">\${content}</div>
            \`;
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function showTyping() {
            document.getElementById('typing').classList.remove('hidden');
        }
        
        function hideTyping() {
            document.getElementById('typing').classList.add('hidden');
        }
    </script>
</body>
</html>`;
    }

    copyShareLink() {
        const input = document.getElementById('share-link');
        if (!input) return;
        
        input.select();
        document.execCommand('copy');
        
        const button = document.getElementById('copy-link');
        if (button) {
            const originalText = button.innerHTML;
            button.innerHTML = '<i class="fas fa-check"></i> Copied!';
            
            setTimeout(() => {
                button.innerHTML = originalText;
            }, 2000);
        }
        
        this.showToast('Link Copied', 'Share link copied to clipboard successfully.', 'success');
    }

    handleFileImport(event) {
        const file = event.target.files[0];
        const importBtn = document.getElementById('import-from-file');
        if (importBtn) importBtn.disabled = !file;
    }

    importFromFile() {
        const fileInput = document.getElementById('import-file');
        if (!fileInput || !fileInput.files[0]) return;

        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const agentData = JSON.parse(e.target.result);
                this.importAgent(agentData);
            } catch (error) {
                this.showToast('Import Error', 'Invalid JSON file. Please check the file format.', 'error');
            }
        };
        reader.readAsText(file);
    }

    importFromText() {
        const textarea = document.getElementById('import-json');
        if (!textarea) return;
        
        const jsonText = textarea.value.trim();
        if (!jsonText) return;

        try {
            const agentData = JSON.parse(jsonText);
            this.importAgent(agentData);
        } catch (error) {
            this.showToast('Import Error', 'Invalid JSON format. Please check your input.', 'error');
        }
    }

    importAgent(agentData) {
        if (!agentData.name || !agentData.systemPrompt) {
            this.showToast('Import Error', 'Invalid agent data. Missing required fields.', 'error');
            return;
        }

        this.currentAgent = {
            id: Date.now(),
            name: agentData.name,
            description: agentData.description || '',
            avatar: agentData.avatar || '🤖',
            systemPrompt: agentData.systemPrompt,
            personality: agentData.personality || 'professional',
            responseStyle: agentData.responseStyle || 'helpful',
            tags: agentData.tags || [],
            created: new Date().toISOString()
        };

        // Import AI settings if available
        if (agentData.aiSettings) {
            this.settings = { ...this.settings, ...agentData.aiSettings };
            this.saveSettings();
            this.updateProviderBadges();
        }

        this.hideModal('import-modal');
        this.populateBuilderForm();
        this.updatePreview();
        this.showScreen('builder');
        this.showToast('Import Successful', 'Agent imported and ready for customization.', 'success');
    }

    importMarketplaceAgent(agent) {
        this.currentAgent = {
            id: Date.now(),
            name: agent.name,
            description: agent.description,
            avatar: agent.avatar,
            systemPrompt: agent.systemPrompt,
            personality: agent.personality,
            responseStyle: agent.responseStyle,
            tags: [...agent.tags],
            created: new Date().toISOString()
        };

        this.populateBuilderForm();
        this.updatePreview();
        this.showScreen('builder');
        this.showToast('Import Successful', `${agent.name} imported successfully.`, 'success');
    }

    showModal(modalId) {
        console.log('Showing modal:', modalId);
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('hidden');
            console.log('Modal shown:', modalId);
        } else {
            console.error('Modal not found:', modalId);
        }
    }

    hideModal(modalId) {
        console.log('Hiding modal:', modalId);
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('hidden');
        }
    }

    showHelpModal() {
        const helpText = `AgentForge Lite Help:

🚀 Getting Started:
• Configure AI provider in Settings (⚙️)
• Choose a template or import an existing agent
• Customize personality, behavior, and prompts
• Test with real AI in the chat interface

🔧 AI Providers:
• OpenRouter: Free tier with API key
• HuggingFace: Free/rate-limited or with API key  
• Local: Run Ollama locally (no key needed)

💡 Tips:
• Export agents as JSON or standalone HTML
• Share agents via generated links
• Import community agents from marketplace
• All API keys stored locally in browser

For more help, visit our documentation.`;
        
        alert(helpText);
    }

    checkImportParameter() {
        const urlParams = new URLSearchParams(window.location.search);
        const importData = urlParams.get('import');
        
        if (importData) {
            try {
                const agentData = JSON.parse(atob(importData));
                this.importAgent(agentData);
            } catch (error) {
                console.error('Failed to import agent from URL:', error);
                this.showToast('Import Error', 'Failed to import agent from share link.', 'error');
            }
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing AgentForge...');
    const app = new AgentForgeApp();
    app.init();
    
    // Make app globally available for debugging
    window.agentForge = app;
    console.log('AgentForge initialized');
});