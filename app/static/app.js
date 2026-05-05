/**
 * app.js
 * Frontend logic for TEKNOFEST AI Platform
 */

const App = {
    state: {
        token: localStorage.getItem('access_token'),
        user: null,
        currentSessionId: null,
        sessions: []
    },

    init() {
        if (this.state.token) {
            this.fetchMe();
        } else {
            UI.showView('auth-view');
        }
    },

    // --- Authentication ---
    async handleAuth(e) {
        e.preventDefault();
        const name = document.getElementById('name').value;
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const isLogin = document.getElementById('tab-login').classList.contains('active');
        const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';

        try {
            // FastAPI OAuth2PasswordRequestForm needs form data URL encoded
            let body;
            let headers = {};
            
            if (isLogin) {
                body = new URLSearchParams();
                body.append('username', email);
                body.append('password', password);
                headers['Content-Type'] = 'application/x-www-form-urlencoded';
            } else {
                body = JSON.stringify({ name, email, password });
                headers['Content-Type'] = 'application/json';
            }

            const res = await fetch(endpoint, { method: 'POST', headers, body });
            if (!res.ok) throw new Error((await res.json()).detail || 'Auth failed');
            
            const data = await res.json();
            this.setToken(data.access_token);
        } catch (error) {
            UI.showError(error.message);
        }
    },

    async guestLogin() {
        try {
            const res = await fetch('/api/auth/guest', { method: 'POST' });
            if (!res.ok) throw new Error('Guest login failed');
            const data = await res.json();
            this.setToken(data.access_token);
        } catch (error) {
            UI.showError(error.message);
        }
    },

    async setToken(token) {
        localStorage.setItem('access_token', token);
        this.state.token = token;
        await this.fetchMe();
    },

    logout() {
        localStorage.removeItem('access_token');
        this.state.token = null;
        this.state.user = null;
        UI.showView('auth-view');
    },

    async fetchMe() {
        if (!this.state.token) return;
        try {
            const res = await fetch('/api/auth/me', {
                headers: { 'Authorization': `Bearer ${this.state.token}` }
            });
            if (res.status === 401) {
                this.logout();
                return;
            }
            this.state.user = await res.json();
            UI.updateUser(this.state.user);
            UI.showView('main-view');
            this.loadSessions();
            
            if (this.state.user.role === 'admin') {
                this.fetchAdminData();
            }
        } catch (err) {
            console.error(err);
            this.logout();
        }
    },

    // --- Chat logic ---
    async loadSessions() {
        try {
            const res = await fetch('/api/chat/sessions', {
                headers: { 'Authorization': `Bearer ${this.state.token}` }
            });
            this.state.sessions = await res.json();
            UI.renderSessionList(this.state.sessions);
            
            if (this.state.sessions.length > 0 && !this.state.currentSessionId) {
                this.selectSession(this.state.sessions[0].id);
            }
        } catch (err) {
            console.error(err);
        }
    },

    async selectSession(sessionId) {
        this.state.currentSessionId = sessionId;
        const session = this.state.sessions.find(s => s.id === sessionId);
        if (session) {
            document.getElementById('current-session-title').textContent = session.title;
        }

        // Highlight sidebar
        document.querySelectorAll('.session-list li').forEach(li => {
            li.classList.toggle('active', li.dataset.id === sessionId);
        });

        // Load messages
        UI.clearMessages();
        UI.showLoadingSkeleton();
        try {
            const res = await fetch(`/api/chat/sessions/${sessionId}/messages`, {
                headers: { 'Authorization': `Bearer ${this.state.token}` }
            });
            const messages = await res.json();
            UI.clearMessages();
            if (messages.length === 0) {
                UI.showEmptyState();
            } else {
                messages.forEach(m => UI.appendMessage(m));
                UI.scrollToBottom();
            }
        } catch (err) {
            console.error(err);
            UI.clearMessages();
        }
    },

    createNewSession() {
        this.state.currentSessionId = null;
        document.getElementById('current-session-title').textContent = "Yeni Sohbet";
        document.querySelectorAll('.session-list li').forEach(li => li.classList.remove('active'));
        UI.clearMessages();
        UI.showEmptyState();
        document.getElementById('message-input').focus();
    },

    async sendMessage(e) {
        e.preventDefault();
        const input = document.getElementById('message-input');
        const text = input.value.trim();
        if (!text) return;

        input.value = '';
        input.style.height = 'auto'; // Reset height

        UI.clearEmptyState();
        UI.appendMessage({ role: 'user', content: text });
        UI.scrollToBottom();

        // Create a temporary AI message with skeleton
        const tempId = 'temp-' + Date.now();
        UI.appendSkeletonMessage(tempId);
        UI.scrollToBottom();

        document.getElementById('send-btn').disabled = true;

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 
                    'Authorization': `Bearer ${this.state.token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: text,
                    session_id: this.state.currentSessionId
                })
            });

            const data = await res.json();
            if (!this.state.currentSessionId) {
                this.state.currentSessionId = data.session_id;
                this.loadSessions(); // refresh sidebar
            }
            
            UI.removeMessage(tempId);
            UI.appendMessage({ role: 'ai', content: data.answer, sources: data.sources });
            UI.scrollToBottom();

        } catch (err) {
            UI.removeMessage(tempId);
            UI.appendMessage({ role: 'ai', content: 'Bir hata oluştu: ' + err.message });
        } finally {
            document.getElementById('send-btn').disabled = false;
        }
    },


    // --- Admin logic ---
    async fetchAdminData() {
        try {
            const [configRes, usersRes, filesRes] = await Promise.all([
                fetch('/api/admin/config', { headers: { 'Authorization': `Bearer ${this.state.token}` } }),
                fetch('/api/admin/users', { headers: { 'Authorization': `Bearer ${this.state.token}` } }),
                fetch('/api/admin/files', { headers: { 'Authorization': `Bearer ${this.state.token}` } })
            ]);

            const config = await configRes.json();
            const users = await usersRes.json();
            const files = await filesRes.json();

            // Populate form
            document.getElementById('llm_provider').value = config.llm_provider;
            
            // Render options based on provider
            UI.updateModelDropdowns(config.llm_provider);

            if (config.llm_model) document.getElementById('llm_model').value = config.llm_model;
            if (config.llm_hallucination_model) document.getElementById('llm_hallucination_model').value = config.llm_hallucination_model;
            if (config.llm_tavily_model) document.getElementById('llm_tavily_model').value = config.llm_tavily_model;

            document.getElementById('retrieval_top_k').value = config.retrieval_top_k;
            document.getElementById('enable_reranking').checked = config.enable_reranking;
            document.getElementById('rag_confidence_threshold').value = config.rag_confidence_threshold;

            // Populate users
            const usersTbody = document.getElementById('users-tbody');
            usersTbody.innerHTML = '';
            users.forEach(u => {
                const mail = u.is_guest ? "Misafir Kullanıcı" : u.email;
                const displayName = u.name ? u.name : "-";
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${u.id}</td>
                    <td>${displayName}</td>
                    <td><span class="badge ${u.role === 'admin' ? 'admin' : ''}">${u.role.toUpperCase()}</span></td>
                    <td>${mail}</td>
                    <td>${u.session_count}</td>
                `;
                usersTbody.appendChild(tr);
            });

            // Populate files
            const filesTbody = document.getElementById('files-tbody');
            filesTbody.innerHTML = '';
            if(files.length === 0) {
                filesTbody.innerHTML = '<tr><td colspan="4" class="text-secondary" style="text-align: center;">Yüklenmiş dosya bulunamadı.</td></tr>';
            } else {
                files.forEach(f => {
                    const kb = (f.size_bytes / 1024).toFixed(1);
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${f.filename}</td>
                        <td>${kb} KB</td>
                        <td>${new Date(f.created_at).toLocaleString('tr-TR')}</td>
                        <td class="text-right">
                            <button class="icon-btn delete-btn" onclick="App.deleteAdminFile('${f.filename}')" title="Sil">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                            </button>
                        </td>
                    `;
                    filesTbody.appendChild(tr);
                });
            }

        } catch (err) {
            console.error("Admin data err", err);
        }
    },

    async saveConfig(e) {
        e.preventDefault();
        const payload = {
            llm_provider: document.getElementById('llm_provider').value,
            llm_model: document.getElementById('llm_model').value,
            llm_hallucination_model: document.getElementById('llm_hallucination_model').value,
            llm_tavily_model: document.getElementById('llm_tavily_model').value,
            retrieval_top_k: parseInt(document.getElementById('retrieval_top_k').value),
            enable_reranking: document.getElementById('enable_reranking').checked,
            rag_confidence_threshold: parseFloat(document.getElementById('rag_confidence_threshold').value)
        };

        const statusLabel = document.getElementById('config-status');
        statusLabel.textContent = "Kaydediliyor...";

        try {
            const res = await fetch('/api/admin/config', {
                method: 'POST',
                headers: { 
                    'Authorization': `Bearer ${this.state.token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            statusLabel.textContent = data.message;
            setTimeout(() => statusLabel.textContent = "", 3000);
        } catch (err) {
            statusLabel.textContent = "Hata oluştu.";
        }
    },

    async runIngestion() {
        const btn = document.getElementById('ingest-btn');
        const out = document.getElementById('ingest-output');
        btn.disabled = true;
        btn.textContent = "Çalıştırılıyor...";
        out.textContent = "İşlem başlatıldı, lütfen bekleyin...\n";

        try {
            const res = await fetch('/api/admin/ingest', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${this.state.token}` }
            });
            const data = await res.json();
            
            if (data.errors) out.textContent += "Errors:\n" + data.errors + "\n";
            if (data.output) out.textContent += "Output:\n" + data.output;
            
        } catch (err) {
            out.textContent += "\nRequest failed: " + err.message;
        } finally {
            btn.disabled = false;
            btn.textContent = "Vektör Veritabanını Güncelle";
        }
    },

    async uploadAdminFile(e) {
        const file = e.target.files[0];
        if(!file) return;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch('/api/admin/files', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${this.state.token}` },
                body: formData
            });
            
            if(res.ok) {
                // Refresh data
                await this.fetchAdminData();
            } else {
                alert("Yükleme başarısız oldu.");
            }
        } catch(err) {
            alert("Upload error: " + err);
        }
        e.target.value = ''; // Reset input
    },

    async deleteAdminFile(filename) {
        if(!confirm(`'${filename}' dosyasını silmek istediğinize emin misiniz?`)) return;

        try {
            const res = await fetch(`/api/admin/files/${encodeURIComponent(filename)}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${this.state.token}` }
            });
            
            if(res.ok) {
                await this.fetchAdminData();
            } else {
                alert("Silme başarısız oldu.");
            }
        } catch(err) {
            alert("Delete error: " + err);
        }
    }
};

const UI = {
    switchAuthTab(tab) {
        document.getElementById('tab-login').classList.remove('active');
        document.getElementById('tab-register').classList.remove('active');
        document.getElementById(`tab-${tab}`).classList.add('active');
        document.getElementById('auth-submit').textContent = tab === 'login' ? 'Giriş Yap' : 'Kayıt Ol';
        
        if (tab === 'register') {
            document.getElementById('name').classList.remove('hidden');
            document.getElementById('name').required = true;
        } else {
            document.getElementById('name').classList.add('hidden');
            document.getElementById('name').required = false;
        }
        
        UI.showError('');
    },

    showView(viewId) {
        document.querySelectorAll('.view-container').forEach(v => v.classList.add('hidden'));
        document.getElementById(viewId).classList.remove('hidden');
    },

    switchMainTab(tab) {
        document.getElementById('chat-area').classList.add('hidden');
        document.getElementById('admin-area').classList.add('hidden');
        document.getElementById(`${tab}-area`).classList.remove('hidden');

        if(tab === 'admin') {
            document.body.classList.add('admin-mode');
        } else {
            document.body.classList.remove('admin-mode');
        }
    },

    showError(msg) {
        document.getElementById('auth-error').textContent = msg;
    },

    toggleTheme() {
        document.body.classList.toggle('dark-mode');
    },

    updateUser(user) {
        const emailLabel = document.getElementById('user-email');
        const avatar = document.getElementById('user-avatar');
        const roleBadge = document.getElementById('user-role');
        
        const displayLabel = user.is_guest ? "Misafir" : (user.name || user.email);
        const avatarChar = user.is_guest ? "M" : (user.name ? user.name.charAt(0).toUpperCase() : user.email.charAt(0).toUpperCase());
        
        emailLabel.textContent = displayLabel;
        avatar.textContent = avatarChar;
        roleBadge.textContent = user.role.toUpperCase();

        if (user.role === 'admin') {
            document.getElementById('admin-btn').classList.remove('hidden');
        } else {
            document.getElementById('admin-btn').classList.add('hidden');
        }
    },

    updateModelDropdowns(provider) {
        if (!provider) return;
        const p = provider.toLowerCase().trim();
        const options = {
            'groq': [
                { id: 'llama-3.3-70b-versatile', label: 'llama-3.3-70b-versatile (Ana Model İçin Tavsiye Edilen)' },
                { id: 'llama-3.1-8b-instant', label: 'llama-3.1-8b-instant (Hızlı & Tavsiye Edilen)' },
                { id: 'mixtral-8x7b-32768', label: 'mixtral-8x7b-32768' },
                { id: 'gemma2-9b-it', label: 'gemma2-9b-it' }
            ],
            'openai': [
                { id: 'gpt-4o-mini', label: 'gpt-4o-mini (Tavsiye Edilen & Fiyat Dostu)' },
                { id: 'gpt-4o', label: 'gpt-4o (Premium)' }
            ],
            'deepseek': [
                { id: 'deepseek-chat', label: 'deepseek-chat (Tavsiye Edilen)' },
                { id: 'deepseek-reasoner', label: 'deepseek-reasoner' }
            ],
            'kimi': [
                { id: 'moonshot-v1-8k', label: 'moonshot-v1-8k (Tavsiye Edilen)' },
                { id: 'moonshot-v1-32k', label: 'moonshot-v1-32k' }
            ]
        };
        const models = options[p] || options['groq'];
        ['llm_model', 'llm_hallucination_model', 'llm_tavily_model'].forEach(id => {
            const select = document.getElementById(id);
            if (!select) return;
            const currentVal = select.value;
            select.innerHTML = '';
            models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.id;
                opt.textContent = m.label;
                select.appendChild(opt);
            });
            
            // Try to keep the previously selected value if it exists in the new list
            if (models.some(m => m.id === currentVal)) {
                select.value = currentVal;
            } else if (models.length > 0) {
                // Otherwise set to the first one
                select.value = models[0].id;
            }
        });
    },

    renderSessionList(sessions) {
        const ul = document.getElementById('session-list');
        ul.innerHTML = '';
        sessions.forEach(s => {
            const li = document.createElement('li');
            li.dataset.id = s.id;
            li.textContent = s.title;
            li.onclick = () => App.selectSession(s.id);
            ul.appendChild(li);
        });
    },

    clearMessages() {
        document.getElementById('messages-container').innerHTML = '';
    },

    showEmptyState() {
        document.getElementById('messages-container').innerHTML = `
            <div class="empty-state">
                <h3>Nasıl Yardımcı Olabilirim?</h3>
                <p>TEKNOFEST 2026 hakkında sorular sorabilirsiniz.</p>
            </div>
        `;
    },

    clearEmptyState() {
        const empty = document.querySelector('.empty-state');
        if (empty) empty.remove();
    },

    appendMessage(msg, id=null) {
        const container = document.getElementById('messages-container');
        const div = document.createElement('div');
        div.className = `message ${msg.role}`;
        if (id) div.id = id;

        const isAi = msg.role === 'ai';
        const avatarStr = isAi ? "AI" : (App.state.user?.is_guest ? "M" : (App.state.user?.email?.[0] || 'U').toUpperCase());
        
        // Parse markdown if marked is available
        const parsedContent = (window.marked && isAi) ? marked.parse(msg.content) : `<p>${msg.content.replace(/\ng/, "<br>")}</p>`;
        
        let sourcesHtml = '';
        if (isAi && msg.sources && msg.sources.length > 0) {
            sourcesHtml = `<div class="sources-box">
                <strong>Kaynaklar:</strong>
                ${msg.sources.map(s => `<span class="source-item">• ${s.metadata?.source || s.type}</span>`).join('')}
            </div>`;
        }

        div.innerHTML = `
            <div class="msg-inner">
                <div class="msg-avatar">${avatarStr}</div>
                <div class="msg-content">
                    ${parsedContent}
                    ${sourcesHtml}
                </div>
            </div>
        `;
        container.appendChild(div);
    },

    appendSkeletonMessage(id) {
        const container = document.getElementById('messages-container');
        const div = document.createElement('div');
        div.className = `message ai`;
        div.id = id;
        div.innerHTML = `
            <div class="msg-inner">
                <div class="msg-avatar">AI</div>
                <div class="msg-content">
                    <div class="skeleton"></div>
                    <div class="skeleton" style="width: 40%"></div>
                </div>
            </div>
        `;
        container.appendChild(div);
    },

    removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    },

    showLoadingSkeleton() {
        this.clearMessages();
        for (let i=0; i<3; i++) {
            this.appendSkeletonMessage(`skel-${i}`);
        }
    },

    scrollToBottom() {
        const c = document.getElementById('messages-container');
        c.scrollTop = c.scrollHeight;
    }
}

// Support cmd+enter or enter to send
document.getElementById('message-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        App.sendMessage(e);
    }
});

// Run Init
window.onload = () => App.init();
