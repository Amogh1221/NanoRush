const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    const innerDiv = document.createElement('div');
    innerDiv.className = 'message-inner';
    
    if (role === 'assistant') {
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar';
        avatarDiv.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10H12V2z"></path><path d="M12 12 2.1 7.1"></path><path d="M12 12l9.9 4.9"></path></svg>';
        innerDiv.appendChild(avatarDiv);
    }
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (role === 'assistant') {
        contentDiv.innerHTML = marked.parse(text || '<span class="typing">...</span>');
    } else {
        contentDiv.textContent = text;
    }
    
    innerDiv.appendChild(contentDiv);
    msgDiv.appendChild(innerDiv);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    
    return contentDiv;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    userInput.value = '';
    
    const assistantContentDiv = appendMessage('assistant', '');
    let fullResponse = "";

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            // Parse SSE format
            const lines = chunk.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    fullResponse += data.chunk;
                    
                    // Strip out any hallucinated User prompt if the stop criteria hasn't caught it yet
                    let cleanResponse = fullResponse.split("User:")[0].split("\nUser")[0];
                    
                    // Render markdown live
                    assistantContentDiv.innerHTML = marked.parse(cleanResponse);
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
            }
        }
    } catch (e) {
        assistantContentDiv.textContent = "Error: Could not connect to the model.";
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
