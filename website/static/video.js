const btnStart  = document.getElementById("btn-start");
const btnStop   = document.getElementById("btn-stop");
const btnReport = document.getElementById("btn-report");
const linkTxt   = document.getElementById("link-txt");
const reportOut = document.getElementById("report-out");

btnStart.onclick = async () => {
  await fetch("/start_stream");
  btnStart.disabled  = true;
  btnStop.disabled   = false;
  btnReport.disabled = true;
  linkTxt.style.display = "none";
};

btnStop.onclick = async () => {
  await fetch("/stop_stream");
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
