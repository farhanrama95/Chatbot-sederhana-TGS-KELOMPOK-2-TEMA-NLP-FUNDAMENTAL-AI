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

function appendOptions(options) {
  if (!options || options.length === 0) return;

  const container = document.createElement("div");
  container.classList.add("options-container");

  options.forEach((opt) => {
    const btn = document.createElement("button");
    btn.classList.add("option-btn");
    btn.textContent = opt.label;
    btn.addEventListener("click", () => {
      sendMessage(opt.value);
    });
    container.appendChild(btn);
  });

  chatMessages.appendChild(container);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage(customMessage) {
  const message = (customMessage ?? userInput.value).trim();
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
    appendOptions(data.options);
  } catch (error) {
    appendMessage("Tidak dapat terhubung ke server.", "bot");
  }
}

sendBtn.addEventListener("click", () => sendMessage());
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

appendMessage("Halo! Saya PuscaBot 🤖. Tanyakan apa saja seputar perpustakaan. berikut pertanyaan yang bisa kamu tanyakan: \n1.jam buka \n2.cara pinjam \n3.batas pinjam \n4.denda \n5.daftar anggota \n6.cari buku \n7.berterimakasih \n8.list buku \n9.tentang", "bot");