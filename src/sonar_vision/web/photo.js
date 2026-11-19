"use strict";
(() => {
  const get=id=>document.getElementById(`photo-${id}`);
  let file=null, url=null, busy=false, job=null, speechText="";
  const ready=()=>{get("start").disabled=busy || !file || !get("permission").checked;};
  function lock(value) {
    busy=value;get("file").disabled=value;get("example").disabled=value;ready();
  }
  async function request(path,options={}) {
    const r=await fetch(path,{...options,signal:AbortSignal.timeout(40000)});
    if(!r.ok){const d=await r.json().catch(()=>({}));throw new Error(typeof d.detail==="string"?d.detail:`Falha HTTP ${r.status}`);}
    return r;
  }
  function select(value) {
    if("speechSynthesis" in window)window.speechSynthesis.cancel();
    file=null;speechText="";get("audio").disabled=true;get("error").textContent="";
    get("rois").replaceChildren();get("annotated").hidden=true;get("report").hidden=true;
    get("metrics").textContent="Nenhuma inferência executada nesta fotografia.";
    if(!value || !/\.(jpe?g|png)$/i.test(value.name) || value.size>12*1024*1024){
      get("error").textContent="Selecione JPEG/PNG de até12MiB.";ready();return;
    }
    file=value;if(url)URL.revokeObjectURL(url);url=URL.createObjectURL(file);
    get("original").src=url;get("original").hidden=false;
    get("state").textContent="Fotografia selecionada. Autorize e execute a inferência nova.";ready();
  }
  get("file").addEventListener("change",()=>select(get("file").files[0]));
  get("permission").addEventListener("change",ready);
  get("example").addEventListener("click",async()=>{
    try {
      const r=await request("/api/video-demo/photo/example");
      const value=new File([await r.blob()],"photograph-real.jpg",{type:"image/jpeg"});
      const transfer=new DataTransfer();transfer.items.add(value);get("file").files=transfer.files;select(value);
    } catch(e) {get("error").textContent=e.message;}
  });
  get("form").addEventListener("submit",async event=>{
    event.preventDefault();if(busy || !file || !get("permission").checked)return;
    lock(true);speechText="";get("audio").disabled=true;get("error").textContent="";
    get("annotated").hidden=true;get("rois").replaceChildren();get("report").hidden=true;
    get("state").textContent="Carregando BR V1 e OCR para inferência nova na fotografia…";
    try {
      const r=await request(`/api/video-demo/photo/jobs?filename=${encodeURIComponent(file.name)}`,{
        method:"POST",headers:{"Content-Type":file.type,"X-Sonar-Media-Permission":"confirmed"},body:file});
      job=(await r.json()).job_id;poll(job);
    } catch(e) {get("error").textContent=e.message;lock(false);}
  });
  async function poll(id) {
    try {
      const data=await(await request(`/api/video-demo/jobs/${id}`)).json();
      if(!["COMPLETED","ERROR","CANCELLED"].includes(data.state)){setTimeout(()=>poll(id),600);return;}
      lock(false);
      if(data.state!=="COMPLETED")throw new Error(data.error || "Análise cancelada");
      const report=data.report,base=`/api/video-demo/photo/jobs/${id}/files/`;
      get("annotated").src=base+"annotated";get("annotated").hidden=false;
      get("report").href=base+"report";get("report").hidden=false;
      const raw=[];get("rois").replaceChildren();
      for(const f of report.findings){
        const card=document.createElement("figure"),img=document.createElement("img"),caption=document.createElement("figcaption");
        img.src=base+`roi-${f.crop_id}`;img.alt="Recorte produzido a partir da caixa prevista pelo YOLO";
        caption.textContent=`ROI prevista ${f.crop_size.join("×")}px · YOLO ${f.detector_confidence.toFixed(3)} · OCR bruto: ${f.readings.map(r=>`${r.text} (score OCR ${r.confidence.toFixed(3)})`).join(" | ") || "sem texto"} · Código candidato: ${f.code || "nenhum — normalização incerta"}`;
        card.append(img,caption);get("rois").append(card);raw.push(...f.readings.map(r=>r.text));
      }
      get("state").textContent=report.state==="NO_PANEL"?"Fotografia: painel não localizado.":"Fotografia: inferência concluída — INCERTO, sem confirmação temporal.";
      get("metrics").textContent=`Modalidade: fotografia estática\nModelo: BR V1 · entrada 640 · OCR E3\nPainéis previstos: ${report.findings.length}\nTempo com modelos: ${report.total_seconds_including_models.toFixed(2)}s\nConfirmação temporal: não validada\nAcurácia: não calculada (sem gabarito na inferência)`;
      const texts=[...new Set(raw)].sort((a,b)=>b.length-a.length);
      speechText=texts.length?`Texto OCR em fotografia estática: ${texts[0]}. Leitura incerta, sem confirmação de linha.`:"";
      get("audio").disabled=!speechText || !("speechSynthesis" in window);
      if(!("speechSynthesis" in window))get("voice-status").textContent="Este navegador não oferece voz; consulte o OCR em texto.";
    } catch(e) {get("error").textContent=e.message;lock(false);speechText="";get("audio").disabled=true;}
  }
  get("audio").addEventListener("click",()=>{
    if(!speechText || !("speechSynthesis" in window))return;
    window.speechSynthesis.cancel();const utterance=new SpeechSynthesisUtterance(speechText);utterance.lang="pt-BR";
    const voice=window.speechSynthesis.getVoices().find(v=>v.lang.toLowerCase()==="pt-br");if(voice)utterance.voice=voice;
    utterance.onerror=()=>{get("voice-status").textContent="Áudio indisponível. O texto OCR permanece visível.";};
    utterance.onend=()=>{get("voice-status").textContent="Leitura do aplicativo encerrada. Não é confirmação de linha.";};
    get("voice-status").textContent="Leitura solicitada ao navegador. Se não houver voz disponível, consulte o texto OCR.";
    window.speechSynthesis.speak(utterance);
  });
  window.addEventListener("pagehide",()=>{
    if(url)URL.revokeObjectURL(url);
    if("speechSynthesis" in window)window.speechSynthesis.cancel();
    if(busy && job)fetch(`/api/video-demo/jobs/${job}`,{method:"DELETE",keepalive:true}).catch(()=>{});
  });
})();
