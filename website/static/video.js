const btnStart  = document.getElementById("btn-start");
const btnStop   = document.getElementById("btn-stop");
const btnReport = document.getElementById("btn-report");
const linkTxt   = document.getElementById("link-txt");
const reportOut = document.getElementById("report-out");

/* ★ añadimos referencias al vídeo */
const videoFeed      = document.getElementById("video-feed");
const videoContainer = document.getElementById("video-container");

btnStart.onclick = async () => {
  await fetch("/start_stream");
  btnStart.disabled  = true;
  btnStop.disabled   = false;
  btnReport.disabled = true;
  linkTxt.style.display = "none";

  /* ★ mostrar y arrancar el stream */
  videoFeed.src = "/video_feed";
  videoFeed.style.display  = "block";
  videoContainer.style.display = "block";
};

btnStop.onclick = async () => {
  await fetch("/stop_stream");

  /* ★ limpiar y ocultar el <img> para que no quede congelado */
  videoFeed.removeAttribute("src");   // borra la última imagen
  videoFeed.style.display  = "none";
  videoContainer.style.display = "none";

  btnStart.disabled  = false;
  btnStop.disabled   = true;
  btnReport.disabled = false;
  linkTxt.style.display = "inline";
};

btnReport.onclick = async () => {
  const r = await fetch("/get_report");
  const data = await r.json();
  reportOut.textContent = JSON.stringify(data, null, 2);
};