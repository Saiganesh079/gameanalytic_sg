const API = "http://localhost:8000"; // Adjust if deployed elsewhere

// Load Genres
fetch(`${API}/genres`)
  .then(res => res.json())
  .then(data => {
    const select = document.getElementById("genre-select");
    data.forEach(g => {
      const opt = document.createElement("option");
      opt.value = g.slug;
      opt.textContent = g.name;
      select.appendChild(opt);
    });
  });

// Load Platforms
fetch(`${API}/platforms`)
  .then(res => res.json())
  .then(data => {
    const select = document.getElementById("platform-select");
    data.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name;
      select.appendChild(opt);
    });
  });

// Load Games & Plot Ratings
function fetchGames() {
  const genres = Array.from(document.getElementById("genre-select").selectedOptions).map(o => o.value);
  const platform = document.getElementById("platform-select").value;
  const start = document.getElementById("start-date").value;
  const end = document.getElementById("end-date").value;

  const params = new URLSearchParams();
  if (genres.length) genres.forEach(g => params.append("genres", g));
  if (platform) params.append("platform", platform);
  if (start && end) {
    params.append("start_date", start);
    params.append("end_date", end);
  }

  fetch(`${API}/games?${params}`)
    .then(res => res.json())
    .then(data => {
      const titles = data.map(g => g.name);
      const ratings = data.map(g => g.rating);

      Plotly.newPlot("rating-chart", [{
        x: titles,
        y: ratings,
        type: "bar"
      }], {
        title: "Top Game Ratings",
        margin: { b: 120 }
      });
    });
}

// Predict Popularity
document.getElementById("prediction-form").addEventListener("submit", e => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const payload = {};
  formData.forEach((v, k) => payload[k] = k.includes("rating") ? parseFloat(v) : parseInt(v) || v);

  fetch(`${API}/predict_popularity`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
    .then(res => res.json())
    .then(data => {
      document.getElementById("prediction-result").textContent =
        `Predicted Popularity Score: ${data.predicted_popularity}`;

      const labels = Object.keys(data.feature_importance);
      const values = Object.values(data.feature_importance);

      Plotly.newPlot("feature-chart", [{
        x: values,
        y: labels,
        type: "bar",
        orientation: "h"
      }], {
        title: "Feature Importance"
      });
    });
});
