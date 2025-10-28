// frontend/app.js
document.getElementById("uploadBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("csvFile");
  const file = fileInput.files[0];
  if (!file) { alert("Please choose a CSV file first."); return; }

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/predict_csv", { method: "POST", body: formData });
  const data = await res.json();
  document.getElementById("result").textContent = JSON.stringify(data, null, 2);
});

document.getElementById("jsonBtn").addEventListener("click", async () => {
  let text = document.getElementById("jsonInput").value;
  if (!text) { alert("Paste a feature JSON first"); return; }
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (e) { alert("Invalid JSON"); return; }

  const res = await fetch("/predict_json", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(parsed)
  });
  const data = await res.json();
  document.getElementById("result").textContent = JSON.stringify(data, null, 2);
});
