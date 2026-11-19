"use strict";
(() => {
  const get = id => document.getElementById(`demo-${id}`);
  const labels = {UPLOADING:"Enviando MP4",STARTING:"Carregando BR V1 e OCR",RUNNING:"Analisando vídeo gravado",ENCODING:"Preparando MP4 para reprodução",COMPLETED:"Análise concluída",CANCELLED:"Análise cancelada",ERROR:"Falha na análise"};
  const finished = new Set(["COMPLETED","ERROR","CANCELLED"]);
  let selected = null, sourceURL = null, previewURL = null, jobID = null, busy = false;
  const ready = () => { get("start").disabled = busy || !selected || !get("permission").checked; };
  const setBusy = value => { busy=value; get("file").disabled=value; get("profile").disabled=value; get("load-example").disabled=value; get("cancel").disabled=!value; ready(); };
  function select(file) {
    get("error").textContent="";
    selected=null;
    if (!file || !/\.mp4$/i.test(file.name) || file.size>64*1024*1024 || file.size<16) {
      get("error").textContent="Selecione um MP4 de até64MiB."; ready(); return;
    }
    selected=file;
    if (sourceURL) URL.revokeObjectURL(sourceURL);
    sourceURL=URL.createObjectURL(file);get("original").src=sourceURL;
    get("original").onloadedmetadata=() => {
      if (!Number.isFinite(get("original").duration) || get("original").duration>60) {
        selected=null; get("error").textContent="Duração máxima:60segundos.";ready();
      }
    };
    get("state").textContent=`Selecionado: ${file.name}. Autorize e inicie a análise.`;ready();
  }
  get("load-example").addEventListener("click", async()=>{
    try {
      const response=await request("/api/video-demo/examples/rotas-approach");
      const file=new File([await response.blob()],"rotas-approach-real.mp4",{type:"video/mp4"});
      const transfer=new DataTransfer();transfer.items.add(file);get("file").files=transfer.files;
      get("profile").value="diagnostic1280";select(file);
    } catch(error) {get("error").textContent=error.message;}
  });
  get("file").addEventListener("change",() => select(get("file").files[0]));
  get("permission").addEventListener("change",ready);
  get("drop").addEventListener("keydown",event => {
    if (!busy && (event.key==="Enter" || event.key===" ")) {event.preventDefault();get("file").click();}
  });
  get("drop").addEventListener("dragover",event => event.preventDefault());
  get("drop").addEventListener("drop",event => {
    event.preventDefault();
    if (!busy && event.dataTransfer.files.length===1) {
      get("file").files=event.dataTransfer.files;select(event.dataTransfer.files[0]);
    }
  });
  async function request(path, options={}) {
    const response=await fetch(path,{...options,signal:AbortSignal.timeout(40000)});
    if (!response.ok) {
      const data=await response.json().catch(() => ({}));
      throw new Error(typeof data.detail==="string" ? data.detail : `Falha HTTP ${response.status}`);
    }
    return response;
  }
  get("form").addEventListener("submit",async event => {
    event.preventDefault();if (!selected || !get("permission").checked || busy) return;
    setBusy(true);get("error").textContent="";get("downloads").hidden=true;
    get("detection-evidence").hidden=true;
    get("result").hidden=true;get("result").removeAttribute("src");get("result").load();
    get("preview").hidden=true;get("progress").value=0;get("readings").replaceChildren();
    get("state").textContent="Enviando MP4 para processamento local…";
    try {
      const response=await request(`/api/video-demo/jobs?filename=${encodeURIComponent(selected.name)}&profile=${encodeURIComponent(get("profile").value)}`,{method:"POST",headers:{"Content-Type":"video/mp4","X-Sonar-Media-Permission":"confirmed"},body:selected});
      const job=await response.json();jobID=job.job_id;poll(jobID);
    } catch(error) {get("error").textContent=error.message;setBusy(false);}
  });
  async function poll(id) {
    if (id!==jobID) return;
    try {
      const job=await (await request(`/api/video-demo/jobs/${id}`)).json();
      get("state").textContent=labels[job.state] || job.state;
      get("progress").value=job.progress || 0;
      const m=job.report || job.metrics;
      if (m) get("metrics").textContent=[
        `Frames amostrados processados: ${m.frames_processed}`,
        `Frames com painel: ${m.frames_with_detection}`,
        `Frames com texto OCR: ${m.frames_with_OCR_text}`,
        `Chamadas OCR: ${m.OCR_calls}`,
        `Frames com código candidato: ${m.frames_with_code_candidate}`,
        `Frames sem código candidato: ${m.uncertain_frames}`,
        `Tempo decorrido: ${job.elapsed_seconds.toFixed(1)} s`,
        m.effective_inference_fps ? `FPS efetivo da análise: ${m.effective_inference_fps.toFixed(2)} (não é acurácia)` : "Modelo em processamento",
        "Confirmação operacional: não validada nesta demonstração"
      ].join("\n");
      if (job.latest) {
        get("readings").replaceChildren();
        for (const finding of job.latest.findings) {
          const item=document.createElement("p");
          item.textContent=`YOLO ${finding.detector_confidence.toFixed(3)} · OCR: ${finding.readings.map(r => `${r.text} (${r.confidence.toFixed(3)})`).join(" | ") || "sem leitura"} · ${finding.code ? `código candidato ${finding.code}` : "leitura incerta"}`;
          get("readings").append(item);
        }
        if (!job.latest.findings.length) get("readings").textContent="Painel não localizado neste frame.";
      }
      if (!finished.has(job.state)) {
        const frame=await request(`/api/video-demo/jobs/${id}/frame`);
        if (frame.status!==204) {
          const url=URL.createObjectURL(await frame.blob());const previous=previewURL;previewURL=url;
          get("preview").onload=()=>{if(previous)URL.revokeObjectURL(previous);};
          get("preview").src=url;get("preview").hidden=false;
        }
        setTimeout(()=>poll(id),600);return;
      }
      setBusy(false);
      if (job.error) get("error").textContent=job.error;
      if (job.state==="COMPLETED") {
        const base=`/api/video-demo/jobs/${id}/files/`;
        get("result").src=base+"video";get("result").hidden=false;get("preview").hidden=true;
        get("save-video").href=base+"video";get("save-report").href=base+"report";
        get("save-frames").href=base+"frames";get("downloads").hidden=false;
        if (m.frames_with_detection>0) {
          get("detection-image").src=base+"detection";get("detection-evidence").hidden=false;
          const first=m.first_detection;
          get("first-reading").textContent=first ? first.findings.map(f=>`YOLO ${f.detector_confidence.toFixed(3)} · OCR executado: ${f.readings.map(r=>r.text).join(" | ") || "sem texto reconhecido"} · ${first.state === "CODE_CANDIDATE" ? "CÓDIGO CANDIDATO" : "INCERTO"}, sem confirmação temporal`).join("\n") : "Consulte as leituras por frame no JSONL.";
        }
      }
    } catch(error) {
      get("error").textContent=error.message;
      fetch(`/api/video-demo/jobs/${id}`,{method:"DELETE"}).catch(()=>{});setBusy(false);
    }
  }
  get("cancel").addEventListener("click",()=>{
    if(jobID) request(`/api/video-demo/jobs/${jobID}`,{method:"DELETE"}).catch(error=>{get("error").textContent=error.message;});
  });
  get("load-static").addEventListener("click",async()=>{
    try {
      const report=await(await request('/api/video-demo/static-evidence/report')).json();
      get("static-image").src='/api/video-demo/static-evidence/image';get("static-image").hidden=false;
      get("static-text").textContent=`Fotografia estática · ${report.author} · ${report.license} · OCR bruto: ${report.findings.flatMap(f=>f.readings.map(r=>r.text)).join(' | ')}. Estado: ${report.state === "CODE_CANDIDATE" ? "código candidato" : "leitura incerta"}, sem confirmação temporal. Inferência registrada, sem caixa manual.`;
    } catch(error) {get("static-text").textContent=error.message;}
  });
  window.addEventListener("pagehide",()=>{
    if(busy && jobID) fetch(`/api/video-demo/jobs/${jobID}`,{method:"DELETE",keepalive:true}).catch(()=>{});
    if(sourceURL)URL.revokeObjectURL(sourceURL);if(previewURL)URL.revokeObjectURL(previewURL);
  });
})();
