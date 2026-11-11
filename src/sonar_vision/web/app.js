"use strict";
const el = (id) => document.getElementById(id);
let sessionId = null;
let active = false;
let lastMessage = "";
let frameUrl = null;
const seen = new Set();
const terminal = new Set(["COMPLETED", "STOPPED", "ERROR"]);
const stateLabels = {STARTING:"Preparando",RUNNING:"Em análise",COMPLETED:"Concluída",STOPPED:"Encerrada",ERROR:"Erro"};
const speechAvailable = "speechSynthesis" in window;
if (!speechAvailable) {
  el("audio").checked = false;
  el("audio").disabled = true;
  el("voice-status").textContent = "Este navegador não oferece síntese de voz. Os resultados continuam disponíveis em texto.";
}
function speak(text) {
  if (!speechAvailable || !el("audio").checked || !text) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "pt-BR";
  const voice = window.speechSynthesis.getVoices().find(v => v.lang.toLowerCase() === "pt-br");
  if (voice) utterance.voice = voice;
  utterance.onerror = () => { el("voice-status").textContent = "A voz não pôde ser reproduzida. Consulte o resultado em texto ou tente Ouvir última informação."; };
  window.speechSynthesis.speak(utterance);
}
async function request(path, options = {}) {
  const response = await fetch(path, {...options, signal: AbortSignal.timeout(12000)});
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = Array.isArray(data.detail) ? data.detail.map(e => e.msg).join("; ") : data.detail;
    throw new Error(detail || `Falha HTTP ${response.status}`);
  }
  return response;
}
function setActive(value) {
  active = value;
  el("start").disabled = value;
  el("stop").disabled = !value;
  el("mode").disabled = value;
  el("route").disabled = value;
  el("url").disabled = value || el("mode").value === "simulation";
}
el("mode").addEventListener("change", () => {
  el("url").disabled = el("mode").value === "simulation";
  el("url").required = el("mode").value === "real";
});
el("url").required = true;
el("session-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  el("error").textContent = "";
  setActive(true);
  el("state").textContent = "Preparando";
  if (speechAvailable) window.speechSynthesis.cancel();
  try {
    const mode = el("mode").value;
    const response = await request("/api/sessions", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({
      route_requested:el("route").value, mode,
      source_url:mode === "real" ? el("url").value : null,
      max_seconds:30, max_frames:150, fps:5
    })});
    const session = await response.json();
    sessionId = session.session_id;
    seen.clear();
    lastMessage = "";
    el("repeat").disabled = true;
    el("events").replaceChildren();
    el("empty-history").hidden = false;
    el("simulation-label").hidden = mode !== "simulation";
    el("result-text").textContent = mode === "simulation" ? "Simulação iniciada. Cenas e leituras artificiais." : "Aguardando leituras de múltiplos frames.";
    el("frame").hidden = true;
    el("empty-frame").hidden = false;
    poll(sessionId);
  } catch (error) {
    el("error").textContent = error.message;
    el("state").textContent = "Não iniciada";
    setActive(false);
  }
});
async function poll(id) {
  if (id !== sessionId) return;
  try {
    const session = await (await request(`/api/sessions/${id}`)).json();
    el("state").textContent = stateLabels[session.state] || session.state;
    el("stats").textContent = `${session.frames_processed} frames processados${session.mode === "simulation" ? " · simulados" : ""}`;
    const fresh = session.events.filter(event => !seen.has(event.event_id));
    for (const event of fresh) {
      seen.add(event.event_id);
      const item = document.createElement("li");
      const time = document.createElement("time");
      time.dateTime = event.timestamp;
      time.textContent = new Date(event.timestamp).toLocaleTimeString("pt-BR");
      const body = document.createElement("div");
      body.textContent = event.announcement;
      const detail = document.createElement("small");
      detail.textContent = `Veículo observado #${event.track_id} · ${event.supporting_frames}/${event.frames_considered} frames de apoio · ${event.match_status}${event.simulated ? " · SIMULAÇÃO" : ""}`;
      body.append(detail);
      item.append(time, body);
      el("events").prepend(item);
      while (el("events").children.length > 100) el("events").lastChild.remove();
      lastMessage = event.announcement;
    }
    if (fresh.length) {
      el("result-text").textContent = lastMessage;
      el("repeat").disabled = !speechAvailable;
      el("empty-history").hidden = true;
      speak(lastMessage);
    }
    el("event-count").textContent = `${seen.size} eventos`;
    const frame = await request(`/api/sessions/${id}/frame`);
    if (frame.status !== 204) {
      const url = URL.createObjectURL(await frame.blob());
      const previousUrl = frameUrl;
      frameUrl = url;
      el("frame").onload = () => { if (previousUrl) URL.revokeObjectURL(previousUrl); };
      el("frame").src = url;
      el("frame").hidden = false;
      el("empty-frame").hidden = true;
    }
    if (session.error) el("error").textContent = session.error;
    if (terminal.has(session.state)) {
      setActive(false);
      if (!session.events.length) {
        lastMessage = "Análise encerrada. Não foi possível confirmar a identificação de um ônibus.";
        el("result-text").textContent = lastMessage;
        el("repeat").disabled = !speechAvailable;
        speak(lastMessage);
      }
      return;
    }
    setTimeout(() => poll(id), 500);
  } catch (error) {
    el("error").textContent = `Conexão interrompida: ${error.message}`;
    // Stop the server session too; a lost browser connection must not leave work unbounded.
    fetch(`/api/sessions/${id}`, {method:"DELETE"}).catch(() => {});
    setActive(false);
  }
}
el("stop").addEventListener("click", async () => {
  if (!sessionId) return;
  el("stop").disabled = true;
  try {
    await request(`/api/sessions/${sessionId}`, {method:"DELETE"});
    el("state").textContent = "Encerrando";
    if (speechAvailable) window.speechSynthesis.cancel();
  } catch (error) { el("error").textContent = error.message; el("stop").disabled = false; }
});
el("repeat").addEventListener("click", () => speak(lastMessage));
window.addEventListener("pagehide", () => {
  if (active && sessionId) fetch(`/api/sessions/${sessionId}`, {method:"DELETE",keepalive:true}).catch(() => {});
  if (frameUrl) URL.revokeObjectURL(frameUrl);
});
request("/health").then(r => r.json()).then(data => {
  el("readiness").textContent = data.custom_model_available
    ? "Pesos encontrados. A qualidade e a procedência precisam ser verificadas no relatório."
    : "Modelo customizado ainda não instalado. A simulação permite explorar a interface.";
}).catch(() => {el("readiness").textContent = "Serviço indisponível.";});
