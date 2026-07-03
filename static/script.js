const chatMessages = document.getElementById("chat-messages");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

function appendMessage(text, sender) {
  const messageEl = document.createElement("div");
  messageEl.classList.add("message", `${sender}-message`);
  messageEl.textContent = text;

  chatMessages.appendChild(messageEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  appendMessage(message, "user");
  userInput.value = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    appendMessage(data.reply || "Terjadi kesalahan pada server.", "bot");
  } catch (error) {
    appendMessage("Tidak dapat terhubung ke server.", "bot");
  }
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

appendMessage("Halo! Saya PuscaBot 🤖. Tanyakan apa saja seputar perpustakaan.", "bot");