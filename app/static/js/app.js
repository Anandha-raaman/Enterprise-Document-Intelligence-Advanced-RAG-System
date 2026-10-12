/**
 * Enterprise Document Intelligence & Advanced RAG System
 * Frontend Client Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // Current Session ID
    const sessionId = 'session_' + Math.random().toString(36).substring(2, 9);
    let activeCitationsMap = {};

    // -------------------------------------------------------------
    // Tab Switching Navigation
    // -------------------------------------------------------------
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            navItems.forEach(n => n.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            item.classList.add('active');
            const targetPane = document.getElementById(targetTab);
            if (targetPane) {
                targetPane.classList.add('active');
            }
        });
    });

    // -------------------------------------------------------------
    // Health & System Info Loader
    // -------------------------------------------------------------
    async function loadSystemHealth() {
        try {
            const resp = await fetch('/api/health');
            if (!resp.ok) return;
            const data = await resp.json();

            const chunks = data.components?.chroma_db?.total_chunks || 0;
            const docCount = data.catalog?.total_documents || 0;

            document.getElementById('header-chunk-count').textContent = chunks;
            document.getElementById('docs-count-pill').textContent = docCount;

            const chromaStatus = document.getElementById('status-chroma');
            if (chromaStatus) {
                chromaStatus.textContent = `ChromaDB (${chunks} chunks)`;
            }

            renderDocumentsTable(data.catalog?.documents || []);
        } catch (err) {
            console.error('Failed to load system health:', err);
        }
    }

    // -------------------------------------------------------------
    // Documents Management & Ingestion Hub
    // -------------------------------------------------------------
    const documentsTableBody = document.getElementById('documents-table-body');
    const refreshDocsBtn = document.getElementById('btn-refresh-docs');
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('pdf-file-input');
    const dropzoneTrigger = document.getElementById('dropzone-trigger');
    const selectedFileLabel = document.getElementById('selected-file-label');
    const taskListContainer = document.getElementById('ingestion-task-list');

    // Trigger file dialog
    dropzoneTrigger.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            selectedFileLabel.textContent = `Selected: ${fileInput.files[0].name} (${(fileInput.files[0].size / 1024).toFixed(1)} KB)`;
        }
    });

    // Upload Form Submit -> Async Queue
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!fileInput.files.length) return;

        const file = fileInput.files[0];
        const customTitle = document.getElementById('doc-custom-title').value.trim();

        const formData = new FormData();
        formData.append('file', file);
        if (customTitle) {
            formData.append('title', customTitle);
        }

        try {
            const submitBtn = document.getElementById('btn-submit-upload');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span>Uploading & Enqueuing...</span>';

            const resp = await fetch('/api/documents/upload', {
                method: 'POST',
                body: formData
            });

            const data = await resp.json();
            if (resp.status === 202) {
                // Task enqueued
                addTaskToMonitor(data.task_id, file.name);
                pollTaskStatus(data.task_id);
                uploadForm.reset();
                selectedFileLabel.textContent = 'PDF format supported (Up to 50MB)';
            } else {
                alert(`Upload failed: ${data.error || 'Unknown error'}`);
            }

            submitBtn.disabled = false;
            submitBtn.innerHTML = '<span>Enqueue for Background Ingestion</span>';
        } catch (err) {
            console.error('Upload error:', err);
            alert('Failed to submit document to ingestion queue.');
            document.getElementById('btn-submit-upload').disabled = false;
        }
    });

    function addTaskToMonitor(taskId, filename) {
        let emptyNotice = taskListContainer.querySelector('.empty-state-notice');
        if (emptyNotice) emptyNotice.remove();

        const taskHtml = `
            <div class="task-item" id="task-${taskId}">
                <div class="task-item-header">
                    <span class="task-filename">${filename}</span>
                    <span class="task-status-badge task-status-indexing" id="badge-${taskId}">QUEUED</span>
                </div>
                <div class="task-progress-bar">
                    <div class="task-progress-fill" id="progress-${taskId}" style="width: 5%;"></div>
                </div>
                <div class="task-stage-desc" id="stage-${taskId}">Enqueued for processing...</div>
            </div>
        `;
        taskListContainer.insertAdjacentHTML('afterbegin', taskHtml);
    }

    function pollTaskStatus(taskId) {
        const interval = setInterval(async () => {
            try {
                const resp = await fetch(`/api/documents/status/${taskId}`);
                if (!resp.ok) {
                    clearInterval(interval);
                    return;
                }
                const task = await resp.json();

                const progressEl = document.getElementById(`progress-${taskId}`);
                const stageEl = document.getElementById(`stage-${taskId}`);
                const badgeEl = document.getElementById(`badge-${taskId}`);

                if (progressEl) progressEl.style.width = `${task.progress}%`;
                if (stageEl) stageEl.textContent = task.stage || task.status;

                if (task.status === 'completed') {
                    clearInterval(interval);
                    if (badgeEl) {
                        badgeEl.className = 'task-status-badge task-status-completed';
                        badgeEl.textContent = 'COMPLETED';
                    }
                    if (stageEl) {
                        const s = task.stats || {};
                        stageEl.textContent = `Completed: ${s.total_chunks || 0} chunks (${s.total_tokens || 0} tokens) across ${s.total_pages || 0} pages.`;
                    }
                    loadSystemHealth();
                } else if (task.status === 'failed') {
                    clearInterval(interval);
                    if (badgeEl) {
                        badgeEl.className = 'task-status-badge task-status-failed';
                        badgeEl.textContent = 'FAILED';
                    }
                    if (stageEl) stageEl.textContent = `Error: ${task.error || 'Failed'}`;
                }
            } catch (err) {
                console.error('Task poll error:', err);
                clearInterval(interval);
            }
        }, 1200);
    }

    // Refresh Documents
    refreshDocsBtn.addEventListener('click', loadSystemHealth);

    function renderDocumentsTable(docs) {
        if (!docs || docs.length === 0) {
            documentsTableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center">No documents indexed yet. Click "Index Sample Docs" above to seed.</td>
                </tr>
            `;
            return;
        }

        documentsTableBody.innerHTML = docs.map(doc => {
            const sectionsBadge = (doc.sections || []).slice(0, 3).map(s => `<span class="badge-tag">${s}</span>`).join(' ') || '<span class="text-muted">General</span>';
            return `
                <tr>
                    <td>
                        <strong style="color: #ffffff;">${doc.doc_title}</strong>
                        <div style="font-size: 11px; color: var(--text-muted);">${doc.file_name}</div>
                    </td>
                    <td>${doc.max_page}</td>
                    <td><strong class="highlight-cyan">${doc.chunk_count}</strong></td>
                    <td>${doc.total_tokens.toLocaleString()}</td>
                    <td>${sectionsBadge}</td>
                    <td>
                        <button class="btn btn-sm btn-outline btn-delete-doc" data-id="${doc.doc_id}" style="color: var(--accent-rose);">
                            Delete
                        </button>
                    </td>
                </tr>
            `;
        }).join('');

        // Attach delete listeners
        document.querySelectorAll('.btn-delete-doc').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const docId = btn.getAttribute('data-id');
                if (confirm(`Remove document '${docId}' from vector & BM25 indexes?`)) {
                    await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
                    loadSystemHealth();
                }
            });
        });
    }

    // Seed Sample Docs Helper
    const seedDocsBtn = document.getElementById('btn-seed-sample-docs');
    seedDocsBtn.addEventListener('click', async () => {
        seedDocsBtn.disabled = true;
        seedDocsBtn.innerHTML = '<span>Seeding Docs...</span>';

        try {
            // Trigger inspection task for TechCorp and NovaCloud sample files
            const sampleFiles = [
                { path: 'sample_docs/TechCorp_Employee_Handbook_2026.pdf', title: 'TechCorp Global Employee Handbook (2026)' },
                { path: 'sample_docs/NovaCloud_Enterprise_Architecture_SLA.pdf', title: 'NovaCloud Enterprise Architecture & SLA (2026)' }
            ];

            // Trigger upload via API or custom endpoint
            for (const sample of sampleFiles) {
                // Fetch the PDF blob from local or server
                const resp = await fetch(`/api/documents/upload`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sample_path: sample.path, title: sample.title })
                }).catch(() => null);
            }

            // Also reload health
            setTimeout(() => {
                loadSystemHealth();
                seedDocsBtn.disabled = false;
                seedDocsBtn.innerHTML = '<span>Sample Docs Seeded</span>';
            }, 1000);
        } catch (err) {
            console.error('Error seeding sample docs:', err);
            seedDocsBtn.disabled = false;
            seedDocsBtn.innerHTML = '<span>Index Sample Docs</span>';
        }
    });

    // -------------------------------------------------------------
    // Intelligence Chat (SSE Streaming & Citations)
    // -------------------------------------------------------------
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input-field');
    const chatContainer = document.getElementById('chat-messages-container');
    const streamCheckbox = document.getElementById('stream-checkbox');
    const clearChatBtn = document.getElementById('btn-clear-chat');
    const guardrailScoreText = document.getElementById('guardrail-score-text');
    const guardrailMeterFill = document.getElementById('guardrail-meter-fill');
    const guardrailStatusPill = document.getElementById('guardrail-status-pill');
    const guardrailDetailsText = document.getElementById('guardrail-details-text');
    const inspectorCitationsList = document.getElementById('inspector-citations-list');

    // Quick prompt chips
    document.querySelectorAll('.quick-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.getAttribute('data-prompt');
            chatInput.value = prompt;
            chatForm.dispatchEvent(new Event('submit'));
        });
    });

    // Clear chat
    clearChatBtn.addEventListener('click', async () => {
        await fetch(`/api/conversations/${sessionId}/clear`, { method: 'POST' });
        chatContainer.innerHTML = `
            <div class="chat-welcome-card">
                <h2>Conversation History Cleared</h2>
                <p>New multi-turn session initialized.</p>
            </div>
        `;
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        chatInput.value = '';
        appendMessage('user', query);

        // Remove welcome card if present
        const welcomeCard = chatContainer.querySelector('.chat-welcome-card');
        if (welcomeCard) welcomeCard.remove();

        const isStream = streamCheckbox.checked;

        if (isStream) {
            await handleStreamingQuery(query);
        } else {
            await handleSyncQuery(query);
        }
    });

    function appendMessage(role, text) {
        const isUser = role === 'user';
        const msgId = 'msg-' + Math.random().toString(36).substring(2, 9);

        const html = `
            <div class="message-bubble ${isUser ? 'message-user' : 'message-assistant'}" id="${msgId}">
                <div class="msg-avatar ${isUser ? 'msg-avatar-user' : 'msg-avatar-ai'}">
                    ${isUser ? 'YOU' : 'AI'}
                </div>
                <div class="msg-content-wrapper">
                    <div class="msg-bubble-box" id="${msgId}-text">
                        ${escapeHtml(text)}
                    </div>
                </div>
            </div>
        `;
        chatContainer.insertAdjacentHTML('beforeend', html);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return { msgId, textEl: document.getElementById(`${msgId}-text`) };
    }

    async function handleStreamingQuery(query) {
        const { msgId, textEl } = appendMessage('assistant', 'Analyzing enterprise documentation...');
        textEl.textContent = '';

        try {
            const resp = await fetch('/api/query/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, session_id: sessionId })
            });

            if (!resp.ok) {
                textEl.textContent = 'Error processing request.';
                return;
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let accumulatedAnswer = '';
            let doneMetadata = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep last incomplete line

                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i].trim();
                    if (line.startsWith('data: ')) {
                        const jsonStr = line.substring(6);
                        try {
                            const parsed = JSON.parse(jsonStr);
                            if (parsed.token) {
                                accumulatedAnswer += parsed.token;
                                textEl.textContent = accumulatedAnswer;
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            } else if (parsed.citations) {
                                doneMetadata = parsed;
                            }
                        } catch (e) {
                            // Non-json chunk
                        }
                    }
                }
            }

            if (doneMetadata) {
                formatFinalResponse(textEl, doneMetadata.answer || accumulatedAnswer, doneMetadata.citations);
                updateGuardrailsUI(doneMetadata.guardrail);
                updateCitationsUI(doneMetadata.citations);
            } else {
                formatFinalResponse(textEl, accumulatedAnswer, []);
            }
        } catch (err) {
            console.error('Streaming error:', err);
            textEl.textContent = 'Connection error during streaming.';
        }
    }

    async function handleSyncQuery(query) {
        const { msgId, textEl } = appendMessage('assistant', 'Searching knowledge base...');

        try {
            const resp = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, session_id: sessionId })
            });

            const data = await resp.json();
            if (resp.ok) {
                formatFinalResponse(textEl, data.answer, data.citations);
                updateGuardrailsUI(data.guardrail);
                updateCitationsUI(data.citations);
            } else {
                textEl.textContent = `Error: ${data.error || 'Query failed'}`;
            }
        } catch (err) {
            console.error('Sync query error:', err);
            textEl.textContent = 'Query processing failed.';
        }
    }

    function formatFinalResponse(containerEl, rawText, citations) {
        activeCitationsMap = {};
        (citations || []).forEach(c => {
            activeCitationsMap[c.ref_id] = c;
        });

        // Replace [REF-1] tags with clickable interactive buttons
        let formatted = escapeHtml(rawText);
        formatted = formatted.replace(/\[(REF-\d+)\]/g, (match, refId) => {
            return `<button class="citation-tag" data-ref="${refId}">${refId}</button>`;
        });

        // Format bullet points and lines
        formatted = formatted.replace(/\n\n/g, '<br><br>').replace(/\n- /g, '<br>&bull; ');

        containerEl.innerHTML = formatted;

        // Attach click listeners to citation tags in response
        containerEl.querySelectorAll('.citation-tag').forEach(tag => {
            tag.addEventListener('click', () => {
                const refId = tag.getAttribute('data-ref');
                if (activeCitationsMap[refId]) {
                    openCitationDrawer(activeCitationsMap[refId]);
                }
            });
        });
    }

    function updateGuardrailsUI(guardrail) {
        if (!guardrail) return;
        const score = guardrail.grounding_percentage || 0;
        guardrailScoreText.textContent = `${score}%`;
        guardrailMeterFill.style.width = `${score}%`;

        guardrailStatusPill.className = 'badge-guardrail';
        if (guardrail.is_grounded) {
            guardrailStatusPill.classList.add('grounded');
            guardrailStatusPill.textContent = 'Verified Grounded';
        } else {
            guardrailStatusPill.classList.add('partial');
            guardrailStatusPill.textContent = guardrail.status || 'Partial Support';
        }

        guardrailDetailsText.textContent = `${guardrail.grounded_claims || 0} of ${guardrail.total_claims || 0} claims strictly grounded. ${guardrail.ungrounded_claims || 0} hallucination flags.`;
    }

    function updateCitationsUI(citations) {
        if (!citations || citations.length === 0) {
            inspectorCitationsList.innerHTML = `<div class="empty-state-notice">No citations referenced in this turn.</div>`;
            return;
        }

        inspectorCitationsList.innerHTML = citations.map(c => `
            <div class="citation-card" data-ref="${c.ref_id}">
                <div class="cit-card-top">
                    <span class="cit-badge">${c.ref_id}</span>
                    <span class="cit-page">Page ${c.page_number}</span>
                </div>
                <div class="cit-title">${c.doc_title}</div>
                <div class="cit-excerpt">${escapeHtml(c.excerpt)}</div>
            </div>
        `).join('');

        // Attach click handlers to inspector cards
        inspectorCitationsList.querySelectorAll('.citation-card').forEach(card => {
            card.addEventListener('click', () => {
                const refId = card.getAttribute('data-ref');
                if (activeCitationsMap[refId]) {
                    openCitationDrawer(activeCitationsMap[refId]);
                }
            });
        });
    }

    // -------------------------------------------------------------
    // Slide-out Citation Detail Drawer
    // -------------------------------------------------------------
    const drawerBackdrop = document.getElementById('citation-drawer-backdrop');
    const drawer = document.getElementById('citation-drawer');
    const btnCloseDrawer = document.getElementById('btn-close-drawer');

    function openCitationDrawer(citation) {
        document.getElementById('drawer-doc-title').textContent = citation.doc_title;
        document.getElementById('drawer-breadcrumb').textContent = citation.hierarchy_path || citation.section;
        document.getElementById('drawer-page-num').textContent = citation.page_number;
        document.getElementById('drawer-section').textContent = citation.section;
        document.getElementById('drawer-relevance').textContent = citation.rerank_score || '0.95';
        document.getElementById('drawer-passage-text').textContent = citation.excerpt;

        drawerBackdrop.classList.add('active');
        drawer.classList.add('active');
    }

    function closeCitationDrawer() {
        drawerBackdrop.classList.remove('active');
        drawer.classList.remove('active');
    }

    btnCloseDrawer.addEventListener('click', closeCitationDrawer);
    drawerBackdrop.addEventListener('click', closeCitationDrawer);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeCitationDrawer();
    });

    // -------------------------------------------------------------
    // TAB 3: Hybrid Retrieval & Reranker Inspector
    // -------------------------------------------------------------
    const inspectForm = document.getElementById('inspect-query-form');
    const inspectInput = document.getElementById('inspect-query-input');
    const inspectSlider = document.getElementById('inspect-alpha-slider');
    const alphaValDisplay = document.getElementById('alpha-val-display');

    inspectSlider.addEventListener('input', () => {
        alphaValDisplay.textContent = inspectSlider.value;
    });

    inspectForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = inspectInput.value.trim();
        if (!query) return;

        const alpha = parseFloat(inspectSlider.value);
        const submitBtn = document.getElementById('btn-run-inspection');
        submitBtn.disabled = true;
        submitBtn.textContent = 'Retrieving & Reranking...';

        try {
            const resp = await fetch('/api/retrieval/inspect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, alpha, top_k: 8 })
            });

            const data = await resp.json();
            if (resp.ok) {
                renderInspectionStages(data);
            }
        } catch (err) {
            console.error('Inspector error:', err);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Run Retrieval Pipeline';
        }
    });

    function renderInspectionStages(data) {
        const denseCardsEl = document.getElementById('stage-dense-cards');
        const sparseCardsEl = document.getElementById('stage-sparse-cards');
        const fusionCardsEl = document.getElementById('stage-fusion-cards');
        const rerankCardsEl = document.getElementById('stage-rerank-cards');

        const denseList = data.hybrid_retrieval?.dense_results || [];
        const sparseList = data.hybrid_retrieval?.sparse_results || [];
        const fusedList = data.hybrid_retrieval?.fused_results || [];
        const rerankedList = data.reranked?.reranked_results || [];

        // Dense
        denseCardsEl.innerHTML = denseList.slice(0, 5).map((item, idx) => `
            <div class="candidate-card">
                <div class="cand-top">
                    <span class="cand-rank">#${idx + 1}</span>
                    <span class="cand-score">Sim: ${(item.dense_score).toFixed(3)}</span>
                </div>
                <div class="cand-text">${escapeHtml(item.text)}</div>
            </div>
        `).join('') || '<div class="empty-state-notice">No dense matches</div>';

        // Sparse
        sparseCardsEl.innerHTML = sparseList.slice(0, 5).map((item, idx) => `
            <div class="candidate-card">
                <div class="cand-top">
                    <span class="cand-rank">#${idx + 1}</span>
                    <span class="cand-score">BM25: ${(item.bm25_score).toFixed(3)}</span>
                </div>
                <div class="cand-text">${escapeHtml(item.text)}</div>
            </div>
        `).join('') || '<div class="empty-state-notice">No sparse matches</div>';

        // Fused
        fusionCardsEl.innerHTML = fusedList.slice(0, 5).map((item, idx) => `
            <div class="candidate-card">
                <div class="cand-top">
                    <span class="cand-rank">#${idx + 1}</span>
                    <span class="cand-score">RRF: ${(item.rrf_score).toFixed(4)}</span>
                </div>
                <div class="cand-text">${escapeHtml(item.text)}</div>
            </div>
        `).join('') || '<div class="empty-state-notice">No fused candidates</div>';

        // Reranked
        rerankCardsEl.innerHTML = rerankedList.map((item, idx) => {
            const delta = item.rank_delta || 0;
            const deltaClass = delta > 0 ? 'positive' : (delta < 0 ? 'negative' : '');
            const deltaStr = delta > 0 ? `+${delta}` : (delta < 0 ? `${delta}` : '=');

            return `
                <div class="candidate-card" style="border-color: rgba(6, 182, 212, 0.4);">
                    <div class="cand-top">
                        <span class="cand-rank highlight-cyan">#${idx + 1}</span>
                        <span class="cand-delta ${deltaClass}">${deltaStr}</span>
                        <span class="cand-score">${(item.rerank_score).toFixed(3)}</span>
                    </div>
                    <div class="cand-text">${escapeHtml(item.text)}</div>
                </div>
            `;
        }).join('') || '<div class="empty-state-notice">No reranked results</div>';
    }

    // -------------------------------------------------------------
    // TAB 4: Ragas Benchmark Studio
    // -------------------------------------------------------------
    const runBenchmarkBtn = document.getElementById('btn-run-benchmark');
    const benchmarkTableBody = document.getElementById('benchmark-cases-body');

    runBenchmarkBtn.addEventListener('click', async () => {
        runBenchmarkBtn.disabled = true;
        runBenchmarkBtn.innerHTML = '<span>Executing Benchmark Suite...</span>';

        try {
            const resp = await fetch('/api/eval/benchmark', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });

            const data = await resp.json();
            if (resp.ok) {
                const metrics = data.summary_metrics || {};
                document.getElementById('metric-lift').textContent = `+${metrics.context_precision_lift_percent}%`;
                document.getElementById('metric-faithfulness').textContent = `${(metrics.faithfulness * 100).toFixed(1)}%`;
                document.getElementById('metric-relevancy').textContent = `${(metrics.answer_relevancy * 100).toFixed(1)}%`;
                document.getElementById('metric-ragas-total').textContent = `${metrics.ragas_score_aggregate}`;

                if (data.detailed_case_evaluations) {
                    benchmarkTableBody.innerHTML = data.detailed_case_evaluations.map(c => `
                        <tr>
                            <td>
                                <strong>${escapeHtml(c.question)}</strong>
                                <div style="font-size: 11px; color: var(--text-muted);">${escapeHtml(c.ground_truth)}</div>
                            </td>
                            <td>${c.context_precision_dense}</td>
                            <td>${c.context_precision_hybrid}</td>
                            <td><span class="badge-success">${c.context_precision_flashrank}</span></td>
                            <td>${c.faithfulness}</td>
                            <td>${c.answer_relevancy}</td>
                        </tr>
                    `).join('');
                }
            }
        } catch (err) {
            console.error('Benchmark error:', err);
        } finally {
            runBenchmarkBtn.disabled = false;
            runBenchmarkBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>Execute Benchmark Suite</span>';
        }
    });

    // Helper: HTML escaping
    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Initial load
    loadSystemHealth();
});
