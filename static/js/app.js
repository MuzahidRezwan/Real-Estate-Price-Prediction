(() => {
  "use strict";

  const state = {
    locations: [],
    cities: [],
    locationCityMap: {},
    locationStats: {},
    numericBounds: {},
    bestModelName: "",
    activeIndex: -1,
    filteredLocations: [],
  };

  const els = {
    form: document.getElementById("prediction-form"),
    locationInput: document.getElementById("location-search"),
    locationListbox: document.getElementById("location-listbox"),
    locationToggle: document.getElementById("location-toggle-btn"),
    locationHint: document.getElementById("location-hint"),
    citySelect: document.getElementById("city-select"),
    areaInput: document.getElementById("floor-area"),
    bedInput: document.getElementById("bedrooms"),
    bathInput: document.getElementById("bathrooms"),
    parkingInput: document.getElementById("parking"),
    areaHint: document.getElementById("area-bounds-hint"),
    bedHint: document.getElementById("bed-bounds-hint"),
    bathHint: document.getElementById("bath-bounds-hint"),
    parkingHint: document.getElementById("parking-bounds-hint"),
    errorBanner: document.getElementById("form-error-banner"),
    predictBtn: document.getElementById("predict-btn"),
    resetBtn: document.getElementById("reset-btn"),
    latencyPill: document.getElementById("latency-pill"),
    placeholder: document.getElementById("result-placeholder"),
    resultContent: document.getElementById("result-content"),
    resBdt: document.getElementById("res-bdt"),
    resLakhCrore: document.getElementById("res-lakh-crore"),
    warningBanner: document.getElementById("warning-banner"),
    warningList: document.getElementById("warning-list"),
    resPps: document.getElementById("res-pps"),
    resModel: document.getElementById("res-model"),
    resLatency: document.getElementById("res-latency"),
    resLocMedian: document.getElementById("res-loc-median"),
    resEcho: document.getElementById("res-echo"),
    metricsTbody: document.getElementById("metrics-tbody"),
    modal: document.getElementById("chart-modal"),
    modalImg: document.getElementById("modal-img"),
    modalTitle: document.getElementById("modal-title"),
    modalClose: document.getElementById("modal-close"),
    modalBackdrop: document.getElementById("modal-backdrop"),
  };

  function formatInt(num) {
    return Number(num).toLocaleString("en-US", { maximumFractionDigits: 0 });
  }

  function formatBdtShort(amount) {
    if (!amount) return "—";
    if (amount >= 1e7) return `${(amount / 1e7).toFixed(2)} Cr BDT`;
    return `${(amount / 1e5).toFixed(2)} Lakh BDT`;
  }

  function showFormError(msg) {
    els.errorBanner.textContent = msg;
    els.errorBanner.classList.remove("hidden");
  }

  function clearFormError() {
    els.errorBanner.textContent = "";
    els.errorBanner.classList.add("hidden");
  }

  function openDropdown() {
    els.locationListbox.classList.remove("hidden");
    els.locationInput.setAttribute("aria-expanded", "true");
  }

  function closeDropdown() {
    els.locationListbox.classList.add("hidden");
    els.locationInput.setAttribute("aria-expanded", "false");
    state.activeIndex = -1;
  }

  function selectLocation(loc) {
    els.locationInput.value = loc;
    closeDropdown();

    const mappedCity = state.locationCityMap[loc];
    if (mappedCity && [...els.citySelect.options].some((o) => o.value === mappedCity)) {
      els.citySelect.value = mappedCity;
    }

    const stats = state.locationStats[loc];
    if (stats) {
      els.locationHint.textContent =
        `${loc}: ${stats.count} cleaned listings · median area ${formatInt(stats.median_area_sft)} sqft · median price ${formatBdtShort(stats.median_price_bdt)}`;
    } else {
      els.locationHint.textContent = "Custom or sparse neighborhood entry.";
    }
  }

  function renderLocationOptions(query = "") {
    const q = query.trim().toLowerCase();
    const matches = state.locations.filter((loc) =>
      loc.toLowerCase().includes(q)
    );
    state.filteredLocations = matches;
    state.activeIndex = matches.length > 0 ? 0 : -1;

    els.locationListbox.innerHTML = "";
    if (matches.length === 0) {
      const li = document.createElement("li");
      li.className = "combobox-empty";
      li.textContent = `No exact match for "${query}" — tip: select "Other" or submit as custom location.`;
      els.locationListbox.appendChild(li);
      return;
    }

    matches.forEach((loc, idx) => {
      const li = document.createElement("li");
      li.className = `combobox-option${idx === state.activeIndex ? " active" : ""}`;
      li.setAttribute("role", "option");
      li.dataset.value = loc;

      const nameSpan = document.createElement("span");
      nameSpan.textContent = loc;

      const metaSpan = document.createElement("span");
      metaSpan.className = "combobox-option-meta";
      const stats = state.locationStats[loc];
      const city = state.locationCityMap[loc] || "";
      metaSpan.textContent = stats ? `${city} (${stats.count})` : city;

      li.appendChild(nameSpan);
      li.appendChild(metaSpan);
      li.addEventListener("mousedown", (e) => {
        e.preventDefault();
        selectLocation(loc);
      });
      els.locationListbox.appendChild(li);
    });
  }

  function updateActiveOptionHighlight() {
    const items = els.locationListbox.querySelectorAll(".combobox-option");
    items.forEach((item, idx) => {
      item.classList.toggle("active", idx === state.activeIndex);
      if (idx === state.activeIndex) {
        item.scrollIntoView({ block: "nearest" });
      }
    });
  }

  function populateCities(cities) {
    els.citySelect.innerHTML = "";
    cities.forEach((city) => {
      const opt = document.createElement("option");
      opt.value = city;
      opt.textContent = city;
      els.citySelect.appendChild(opt);
    });
    if (cities.includes("Dhaka")) {
      els.citySelect.value = "Dhaka";
    }
  }

  function updateBoundsHints(bounds) {
    if (bounds.Floor_area_sft) {
      els.areaHint.textContent = `Training range: ${formatInt(bounds.Floor_area_sft.min)} – ${formatInt(bounds.Floor_area_sft.max)} sqft`;
    }
    if (bounds.Bedrooms) {
      els.bedHint.textContent = `Range: ${bounds.Bedrooms.min} – ${bounds.Bedrooms.max}`;
    }
    if (bounds.Bathrooms) {
      els.bathHint.textContent = `Range: ${bounds.Bathrooms.min} – ${bounds.Bathrooms.max}`;
    }
    if (bounds.Parking_Count) {
      els.parkingHint.textContent = `Range: ${bounds.Parking_Count.min} – ${bounds.Parking_Count.max}`;
    }
  }

  function renderMetricsTable(data) {
    els.metricsTbody.innerHTML = "";
    const bestName = data.best_model_name;
    (data.models_comparison || []).forEach((row) => {
      const tr = document.createElement("tr");
      const isWinner = row.name === bestName;
      if (isWinner) tr.className = "best-row";

      const cells = [
        row.name,
        row.r2.toFixed(4),
        row.cv_r2.toFixed(4),
        `৳ ${formatInt(row.mae)}`,
        `${row.mae_pct_of_mean.toFixed(2)}%`,
        `৳ ${formatInt(row.rmse)}`,
      ];

      cells.forEach((val, idx) => {
        const td = document.createElement("td");
        td.textContent = val;
        if (idx === 0) td.style.fontWeight = "600";
        tr.appendChild(td);
      });

      const statusTd = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = `status-badge ${isWinner ? "winner" : "candidate"}`;
      badge.textContent = isWinner ? "Active Best" : "Benchmarked";
      statusTd.appendChild(badge);
      tr.appendChild(statusTd);

      els.metricsTbody.appendChild(tr);
    });
  }

  async function initDashboard() {
    try {
      const [locRes, metRes] = await Promise.all([
        fetch("/api/locations"),
        fetch("/api/metrics"),
      ]);
      if (locRes.ok) {
        const locData = await locRes.json();
        state.locations = locData.locations || [];
        state.cities = locData.cities || [];
        state.locationCityMap = locData.location_city_map || {};
        state.locationStats = locData.location_stats || {};
        state.numericBounds = locData.numeric_bounds || {};

        populateCities(state.cities);
        updateBoundsHints(state.numericBounds);
        renderLocationOptions("");

        const defaultLoc = state.locations.includes("Dhanmondi")
          ? "Dhanmondi"
          : state.locations[0] || "";
        if (defaultLoc) {
          selectLocation(defaultLoc);
        }
      }
      if (metRes.ok) {
        const metData = await metRes.json();
        state.bestModelName = metData.best_model_name || "";
        renderMetricsTable(metData);
      }
    } catch (err) {
      showFormError("Could not load model metadata from server. Check API availability.");
    }
  }

  function setLoading(isLoading) {
    els.predictBtn.disabled = isLoading;
    const spinner = els.predictBtn.querySelector(".btn-spinner");
    const label = els.predictBtn.querySelector(".btn-label");
    spinner.classList.toggle("hidden", !isLoading);
    label.textContent = isLoading ? "Computing Valuation..." : "Estimate Property Price";
  }

  function renderPrediction(result) {
    els.placeholder.classList.add("hidden");
    els.resultContent.classList.remove("hidden");

    els.resBdt.textContent = result.formatted_bdt;
    els.resLakhCrore.textContent = result.lakh_crore;
    els.resPps.textContent = result.formatted_price_per_sft;
    els.resModel.textContent = result.model_used;
    els.resLatency.textContent = `${result.latency_ms.toFixed(2)} ms`;

    els.latencyPill.textContent = `${result.latency_ms.toFixed(1)} ms`;
    els.latencyPill.classList.add("active");

    const loc = result.normalized_input.location;
    const locStat = state.locationStats[loc];
    els.resLocMedian.textContent = locStat
      ? formatBdtShort(locStat.median_price_bdt)
      : "N/A";

    if (result.out_of_bounds && Array.isArray(result.warnings) && result.warnings.length > 0) {
      els.warningList.innerHTML = "";
      result.warnings.forEach((w) => {
        const li = document.createElement("li");
        li.textContent = w;
        els.warningList.appendChild(li);
      });
      els.warningBanner.classList.remove("hidden");
    } else {
      els.warningBanner.classList.add("hidden");
    }

    const inp = result.normalized_input;
    els.resEcho.textContent =
      `Evaluated Specs: ${inp.location} (${inp.city}) · ${formatInt(inp.floor_area_sft)} sqft · ` +
      `${inp.bedrooms} Bed · ${inp.bathrooms} Bath · ${inp.parking_count} Parking`;
  }

  // Event Listeners
  els.locationInput.addEventListener("focus", () => {
    renderLocationOptions("");
    openDropdown();
  });

  els.locationInput.addEventListener("input", (e) => {
    renderLocationOptions(e.target.value);
    openDropdown();
  });

  els.locationToggle.addEventListener("click", () => {
    if (els.locationListbox.classList.contains("hidden")) {
      renderLocationOptions("");
      openDropdown();
      els.locationInput.focus();
    } else {
      closeDropdown();
    }
  });

  els.locationInput.addEventListener("keydown", (e) => {
    const count = state.filteredLocations.length;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (els.locationListbox.classList.contains("hidden")) {
        renderLocationOptions(els.locationInput.value);
        openDropdown();
      } else if (count > 0) {
        state.activeIndex = (state.activeIndex + 1) % count;
        updateActiveOptionHighlight();
      }
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (count > 0) {
        state.activeIndex = (state.activeIndex - 1 + count) % count;
        updateActiveOptionHighlight();
      }
    } else if (e.key === "Enter" && !els.locationListbox.classList.contains("hidden")) {
      if (state.activeIndex >= 0 && state.activeIndex < count) {
        e.preventDefault();
        selectLocation(state.filteredLocations[state.activeIndex]);
      }
    } else if (e.key === "Escape") {
      closeDropdown();
    }
  });

  document.addEventListener("click", (e) => {
    if (!document.getElementById("location-combobox").contains(e.target)) {
      closeDropdown();
    }
  });

  document.querySelectorAll(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      els.areaInput.value = btn.dataset.area;
    });
  });

  els.resetBtn.addEventListener("click", () => {
    clearFormError();
    const defaultLoc = state.locations.includes("Dhanmondi")
      ? "Dhanmondi"
      : state.locations[0] || "";
    if (defaultLoc) selectLocation(defaultLoc);
    els.areaInput.value = 1350;
    els.bedInput.value = 3;
    els.bathInput.value = 3;
    els.parkingInput.value = 1;
  });

  els.form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearFormError();

    const location = els.locationInput.value.trim();
    const city = els.citySelect.value.trim();
    const floorArea = Number(els.areaInput.value);
    const bedrooms = Number(els.bedInput.value);
    const bathrooms = Number(els.bathInput.value);
    const parkingCount = Number(els.parkingInput.value);

    if (!location) {
      showFormError("Please choose or enter a property location.");
      return;
    }
    if (!Number.isFinite(floorArea) || floorArea <= 0) {
      showFormError("Floor area must be a positive number greater than 0.");
      return;
    }
    if (!Number.isFinite(bedrooms) || bedrooms <= 0) {
      showFormError("Bedrooms must be greater than 0.");
      return;
    }
    if (!Number.isFinite(bathrooms) || bathrooms <= 0) {
      showFormError("Bathrooms must be greater than 0.");
      return;
    }
    if (!Number.isFinite(parkingCount) || parkingCount < 0) {
      showFormError("Parking count cannot be negative.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          location,
          city,
          floor_area_sft: floorArea,
          bedrooms,
          bathrooms,
          parking_count: parkingCount,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        showFormError(data.detail || "Invalid input parameters.");
        return;
      }
      renderPrediction(data);
    } catch (err) {
      showFormError("Prediction request failed. Please verify the backend server is running.");
    } finally {
      setLoading(false);
    }
  });

  // Chart Modal Preview
  document.querySelectorAll(".chart-card").forEach((card) => {
    card.addEventListener("click", () => {
      const img = card.querySelector("img");
      els.modalImg.src = img.src;
      els.modalImg.alt = img.alt;
      els.modalTitle.textContent = card.dataset.chartTitle || "EDA Chart";
      els.modal.classList.remove("hidden");
    });
  });

  function closeModal() {
    els.modal.classList.add("hidden");
    els.modalImg.src = "";
  }

  els.modalClose.addEventListener("click", closeModal);
  els.modalBackdrop.addEventListener("click", closeModal);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.modal.classList.contains("hidden")) {
      closeModal();
    }
  });

  initDashboard();
})();
