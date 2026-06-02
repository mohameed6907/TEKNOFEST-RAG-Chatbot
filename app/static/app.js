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
        this.state.currentSessionId = null;
        this.state.sessions = [];
        
        // Reset UI Views & Tabs
        UI.switchMainTab('chat');
        UI.clearMessages();
        
        const sessionList = document.getElementById('session-list');
        if (sessionList) sessionList.innerHTML = '';
        
        const sessionTitle = document.getElementById('current-session-title');
        if (sessionTitle) sessionTitle.textContent = "Yeni Sohbet";
        
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

            if (!res.ok) {
                // API hata döndürdü (4xx / 5xx) — data.answer undefined olur, marked.parse patlar
                const errMsg = data?.detail || `Sunucu hatası (${res.status})`;
                UI.removeMessage(tempId);
                UI.appendMessage({ role: 'ai', content: `⚠️ ${errMsg}` });
                UI.scrollToBottom();
                return;
            }

            if (!this.state.currentSessionId) {
                this.state.currentSessionId = data.session_id;
                this.loadSessions(); // refresh sidebar
            }
            
            UI.removeMessage(tempId);
            UI.appendMessage({ role: 'ai', content: data.answer ?? '', sources: data.sources });
            UI.scrollToBottom();

        } catch (err) {
            UI.removeMessage(tempId);
            UI.appendMessage({ role: 'ai', content: 'Bir hata oluştu: ' + err.message });
        } finally {
            document.getElementById('send-btn').disabled = false;
        }
    },

    async editSession(id, currentTitle) {
        const newTitle = prompt("Sohbetin yeni başlığını girin:", currentTitle);
        if (!newTitle || newTitle === currentTitle) return;

        try {
            const res = await fetch(`/api/chat/sessions/${id}`, {
                method: 'PUT',
                headers: { 
                    'Authorization': `Bearer ${this.state.token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ title: newTitle })
            });
            if (res.ok) {
                this.loadSessions();
                if (this.state.currentSessionId === id) {
                    document.getElementById('current-session-title').textContent = newTitle;
                }
            } else {
                alert("Başlık güncellenirken hata oluştu.");
            }
        } catch (err) {
            console.error(err);
        }
    },

    async deleteSession(id) {
        if (!confirm("Bu sohbeti silmek istediğinize emin misiniz?")) return;

        try {
            const res = await fetch(`/api/chat/sessions/${id}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${this.state.token}` }
            });
            if (res.ok) {
                if (this.state.currentSessionId === id) {
                    this.createNewSession();
                }
                this.loadSessions();
            } else {
                alert("Sohbet silinirken hata oluştu.");
            }
        } catch (err) {
            console.error(err);
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

            // Populate form — per-purpose provider + model
            document.getElementById('llm_provider').value = config.llm_provider;
            UI.updateModelDropdown('llm_model', config.llm_provider);
            if (config.llm_model) document.getElementById('llm_model').value = config.llm_model;

            const halProv = config.llm_hallucination_provider || config.llm_provider;
            document.getElementById('llm_hallucination_provider').value = halProv;
            UI.updateModelDropdown('llm_hallucination_model', halProv);
            if (config.llm_hallucination_model) document.getElementById('llm_hallucination_model').value = config.llm_hallucination_model;

            const tavProv = config.llm_tavily_provider || config.llm_provider;
            document.getElementById('llm_tavily_provider').value = tavProv;
            UI.updateModelDropdown('llm_tavily_model', tavProv);
            if (config.llm_tavily_model) document.getElementById('llm_tavily_model').value = config.llm_tavily_model;

            const rerProv = config.llm_reranker_provider || config.llm_provider;
            document.getElementById('llm_reranker_provider').value = rerProv;
            UI.updateModelDropdown('llm_reranker_model', rerProv);
            if (config.llm_reranker_model) document.getElementById('llm_reranker_model').value = config.llm_reranker_model;

            if (config.embedding_provider) document.getElementById('embedding_provider').value = config.embedding_provider;
            if (config.embedding_model_name) document.getElementById('embedding_model_name').value = config.embedding_model_name;

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
            llm_hallucination_provider: document.getElementById('llm_hallucination_provider').value,
            llm_hallucination_model: document.getElementById('llm_hallucination_model').value,
            llm_tavily_provider: document.getElementById('llm_tavily_provider').value,
            llm_tavily_model: document.getElementById('llm_tavily_model').value,
            llm_reranker_provider: document.getElementById('llm_reranker_provider').value,
            llm_reranker_model: document.getElementById('llm_reranker_model').value,
            embedding_provider: document.getElementById('embedding_provider').value,
            embedding_model_name: document.getElementById('embedding_model_name').value,
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

    toggleSidebar() {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) {
            sidebar.classList.toggle('collapsed');
        }
    },

    toggleProfileMenu() {
        const menu = document.getElementById('profile-menu');
        if (menu) {
            menu.classList.toggle('hidden');
        }
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

    updateModelDropdown(selectId, provider) {
        if (!provider) return;
        const p = provider.toLowerCase().trim();
        const options = {
            'groq': [
                { id: 'llama-3.3-70b-versatile', label: 'llama-3.3-70b-versatile' },
                { id: 'mixtral-8x7b-32768', label: 'mixtral-8x7b-32768' },
            ],
            'openai': [
                { id: 'gpt-4o-mini', label: 'gpt-4o-mini' },
                { id: 'gpt-4o', label: 'gpt-4o' },
            ],
            'deepseek': [
                { id: 'deepseek-chat', label: 'deepseek-chat' },
            ],
            'kimi': [
                { id: 'moonshot-v1-8k', label: 'moonshot-v1-8k' },
                { id: 'moonshot-v1-32k', label: 'moonshot-v1-32k' },
            ],
            'anthropic': [
                { id: 'claude-3-haiku-20240307', label: 'claude-3-haiku' },
                { id: 'claude-3-5-sonnet-20240620', label: 'claude-3.5-sonnet' },
            ],
            'together': [
                { id: 'meta-llama/Llama-3-70b-chat-hf', label: 'Llama-3-70b-chat' },
                { id: 'mistralai/Mixtral-8x7B-Instruct-v0.1', label: 'Mixtral-8x7B-Instruct' },
            ]
        };
        const models = options[p] || options['openai'];
        const select = document.getElementById(selectId);
        if (!select) return;
        const currentVal = select.value;
        select.innerHTML = '';
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = m.label;
            select.appendChild(opt);
        });
        if (models.some(m => m.id === currentVal)) {
            select.value = currentVal;
        } else if (models.length > 0) {
            select.value = models[0].id;
        }
    },

    // Legacy compat — eski çağrılar için
    updateModelDropdowns(provider) {
        this.updateModelDropdown('llm_model', provider);
        this.updateModelDropdown('llm_hallucination_model', provider);
        this.updateModelDropdown('llm_tavily_model', provider);
        this.updateModelDropdown('llm_reranker_model', provider);
    },

    renderSessionList(sessions) {
        const ul = document.getElementById('session-list');
        ul.innerHTML = '';
        sessions.forEach(s => {
            const li = document.createElement('li');
            li.dataset.id = s.id;
            
            const titleSpan = document.createElement('span');
            titleSpan.className = 'session-title-text';
            titleSpan.textContent = s.title;
            titleSpan.onclick = () => App.selectSession(s.id);
            
            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'session-actions';
            
            const editBtn = document.createElement('button');
            editBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>';
            editBtn.title = "Düzenle";
            editBtn.onclick = (e) => { e.stopPropagation(); App.editSession(s.id, s.title); };
            
            const deleteBtn = document.createElement('button');
            deleteBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>';
            deleteBtn.title = "Sil";
            deleteBtn.onclick = (e) => { e.stopPropagation(); App.deleteSession(s.id); };
            
            actionsDiv.appendChild(editBtn);
            actionsDiv.appendChild(deleteBtn);
            
            li.appendChild(titleSpan);
            li.appendChild(actionsDiv);
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
        // Guard: msg.content null/undefined ise marked.parse "input parameter is undefined" fırlatır
        const safeContent = msg.content ?? '';
        const parsedContent = (window.marked && isAi && safeContent)
            ? marked.parse(safeContent)
            : `<p>${(safeContent).replace(/\n/g, "<br>")}</p>`;

        
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
                    <div class="rocket-loader">
                        <svg class="rocket-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"></path>
                            <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"></path>
                            <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"></path>
                            <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"></path>
                        </svg>
                    </div>
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
