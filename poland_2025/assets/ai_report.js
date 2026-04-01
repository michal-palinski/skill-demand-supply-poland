(function () {
  const payload = window.__AI_REGIONAL_REPORT__;
  const regionalPayload = window.__REGIONAL_REPORT__;

  if (!payload || !window.Plotly || !regionalPayload || !regionalPayload.geojson) {
    return;
  }

  const rows = [...payload.rows];
  const meta = payload.meta || {};
  const geojson = regionalPayload.geojson;

  const fmtInt = (value) =>
    new Intl.NumberFormat("pl-PL", { maximumFractionDigits: 0 }).format(value);

  const fmtRate = (value) =>
    new Intl.NumberFormat("pl-PL", {
      minimumFractionDigits: value < 10 ? 2 : 1,
      maximumFractionDigits: value < 10 ? 2 : 1,
    }).format(value);

  const fmtPct = (value) =>
    new Intl.NumberFormat("pl-PL", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);

  const titleVoiv = (name) =>
    name
      .split("-")
      .map((part) => part.charAt(0).toLocaleUpperCase("pl-PL") + part.slice(1))
      .join("-");

  const displayAiLabel = (label) =>
    String(label).toLowerCase() === "optical character recognition software"
      ? "image recognition"
      : label;

  function renderDetail(row) {
    const detailEl = document.getElementById("ai-regional-detail");
    if (!detailEl || !row) {
      return;
    }

    const chips = (row.top_characteristic_skills || [])
      .map(
        (skill, idx) => `
        <div class="ai-detail-skill">
          <div class="ai-detail-skill-rank">${idx + 1}</div>
          <div>
            <div class="ai-detail-skill-label">${displayAiLabel(skill.label)}</div>
            <div class="ai-detail-skill-meta">${skill.code}</div>
          </div>
        </div>
      `
      )
      .join("");

    detailEl.innerHTML = `
      <div class="ai-detail-card">
        <p class="section-kicker">Selected voivodeship</p>
        <h3>${titleVoiv(row.voivodeship)}</h3>
        <div class="ai-detail-metrics">
          <div class="ai-detail-metric">
            <span class="ai-detail-metric-label">AI offers per 100k LF</span>
            <span class="ai-detail-metric-value">${fmtRate(row.ai_offers_per_100k_lf)}</span>
          </div>
          <div class="ai-detail-metric">
            <span class="ai-detail-metric-label">AI offers</span>
            <span class="ai-detail-metric-value">${fmtInt(row.ai_offers)}</span>
          </div>
          <div class="ai-detail-metric">
            <span class="ai-detail-metric-label">Share of regional offers</span>
            <span class="ai-detail-metric-value">${fmtPct(row.ai_offer_share_pct)}%</span>
          </div>
        </div>
        <div class="ai-detail-skills-head">Top 3 characteristic AI skills by G²</div>
        <div class="ai-detail-skills">${chips}</div>
      </div>
    `;
  }

  function renderMap() {
    const mapEl = document.getElementById("ai-regional-map");
    if (!mapEl) {
      return;
    }

    const locations = rows.map((row) => row.voivodeship);
    Plotly.newPlot(
      mapEl,
      [
        {
          type: "choropleth",
          geojson,
          featureidkey: "properties.nazwa",
          locations,
          z: rows.map((row) => row.ai_offers_per_100k_lf),
          colorscale: [
            [0.0, "#fff2e8"],
            [0.35, "#fdba74"],
            [0.7, "#f97316"],
            [1.0, "#9a3412"],
          ],
          marker: {
            line: { color: "rgba(255,255,255,0.95)", width: 1 },
          },
          colorbar: {
            title: { text: "AI offers<br>per 100k LF" },
            thickness: 16,
            len: 0.7,
            x: 1.03,
            y: 0.5,
            outlinewidth: 0,
          },
          hovertemplate:
            "<b>%{location}</b><extra></extra>",
        },
      ],
      {
        margin: { l: 0, r: 12, t: 0, b: 0 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        dragmode: false,
        showlegend: false,
        geo: {
          fitbounds: "locations",
          visible: false,
          projection: { type: "mercator" },
          bgcolor: "rgba(0,0,0,0)",
        },
      },
      {
        displayModeBar: false,
        responsive: true,
      }
    );

    mapEl.on("plotly_click", (event) => {
      const point = event?.points?.[0];
      if (!point) {
        return;
      }
      const row = rows.find((item) => item.voivodeship === point.location);
      renderDetail(row);
    });

    document.getElementById("ai-map-figure-title").textContent =
      "Figure 3. Regional AI demand intensity across Polish voivodeships";
    document.getElementById("ai-map-note").innerHTML = `
      <div class="figure-caption-source">
        Source: <a href="https://www.pracuj.pl/" target="_blank" rel="noopener noreferrer">Pracuj.pl</a>;
        ESCO; Lightcast AI skills taxonomy; authors' Lightcast-to-ESCO bridge; labour force denominator from
        <a href="https://api.stat.gov.pl/Home/BdlApi?gt=&lang=en" target="_blank" rel="noopener noreferrer">GUS BDL API</a>.
        Regional assignment is available for ${fmtInt(meta.ai_offers_total_mapped)} of ${fmtInt(meta.ai_offers_total)} AI postings.
      </div>
    `;
  }

  renderMap();
  renderDetail(rows[0]);
})();
