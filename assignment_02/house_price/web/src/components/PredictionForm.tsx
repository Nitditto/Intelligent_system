import React, { useState } from "react";
import { PRESETS, type Preset } from "../constants/options";
import type { ListingInput } from "../types";

interface PredictionFormProps {
  formData: ListingInput;
  onFormDataChange: React.Dispatch<React.SetStateAction<ListingInput>>;
  onSubmit: (data: ListingInput) => void;
  loading: boolean;
  onReset: () => void;
  currentStep: number;
  onStepChange: (step: number) => void;
}

const TOP_PROVINCES = [
  { slug: "an-giang", label: "An Giang" },
  { slug: "tp-ho-chi-minh", label: "Ho Chi Minh City" },
  { slug: "ha-noi", label: "Hanoi" },
  { slug: "da-nang", label: "Da Nang" },
  { slug: "binh-duong", label: "Binh Duong" },
  { slug: "dong-nai", label: "Dong Nai" },
  { slug: "can-tho", label: "Can Tho" },
  { slug: "hai-phong", label: "Hai Phong" },
  { slug: "khanh-hoa", label: "Khanh Hoa" },
  { slug: "lam-dong", label: "Lam Dong" },
  { slug: "ba-ria-vung-tau", label: "Ba Ria - Vung Tau" },
  { slug: "long-an", label: "Long An" },
];

const PROPERTY_TYPE_CARDS = [
  { value: "Nhà riêng", title: "Townhouse", desc: "Standalone residential home" },
  { value: "Căn hộ chung cư", title: "Apartment", desc: "Condo / High-rise unit" },
  { value: "Đất", title: "Residential Land", desc: "Plot with building rights" },
  { value: "Nhà mặt phố", title: "Street-Front", desc: "Commercial street access" },
  { value: "Biệt thự", title: "Villa / Estate", desc: "Luxury detached residence" },
];

const POSITION_CARDS = [
  { value: "Đường chính", title: "Main Street", desc: "Direct street front" },
  { value: "Trong hẻm", title: "In Alley", desc: "Quiet residential lane" },
  { value: "Mặt tiền", title: "Corner Frontage", desc: "High visibility corner" },
];

const ROAD_TYPE_CARDS = [
  { value: "Đường nhựa", title: "Asphalt Road", desc: "Smooth paved road" },
  { value: "Đường bê tông", title: "Concrete Road", desc: "Paved concrete alley" },
  { value: "Đường đất", title: "Dirt Road", desc: "Unpaved access" },
];

const DIRECTION_CARDS = [
  { value: "Nam", label: "South" },
  { value: "Đông Nam", label: "Southeast" },
  { value: "Đông", label: "East" },
  { value: "Bắc", label: "North" },
  { value: "Tây", label: "West" },
  { value: "Tây Nam", label: "Southwest" },
  { value: "Tây Bắc", label: "Northwest" },
  { value: "Đông Bắc", label: "Northeast" },
];

const AREA_QUICK_CARDS = [
  { area: 45, label: "Compact", sub: "45 m²" },
  { area: 78.7, label: "Townhouse", sub: "78.7 m²" },
  { area: 120, label: "Spacious", sub: "120 m²" },
  { area: 250, label: "Estate", sub: "250 m²" },
];

const LAYOUT_PRESET_CARDS = [
  { beds: 1, baths: 1, floors: 1, label: "1 Bed Studio" },
  { beds: 2, baths: 2, floors: 1, label: "2 Bed Standard" },
  { beds: 3, baths: 2, floors: 2, label: "3 Bed Family" },
  { beds: 4, baths: 3, floors: 3, label: "4+ Bed Villa" },
];

export const PredictionForm: React.FC<PredictionFormProps> = ({
  formData,
  onFormDataChange,
  onSubmit,
  loading,
  onReset,
  currentStep,
  onStepChange,
}) => {
  const [activePresetId, setActivePresetId] = useState<string>("rach-gia-full");
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleApplyPreset = (preset: Preset) => {
    setActivePresetId(preset.id);
    onFormDataChange(preset.data);
    setValidationError(null);
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
    setActivePresetId("");

    if (type === "number") {
      const parsed = value === "" ? undefined : parseFloat(value);
      onFormDataChange((prev) => ({ ...prev, [name]: parsed }));
    } else {
      onFormDataChange((prev) => ({ ...prev, [name]: value === "" ? undefined : value }));
    }
  };

  const updateCounter = (field: "Bedrooms" | "Bathrooms" | "Floors", delta: number) => {
    setActivePresetId("");
    onFormDataChange((prev) => {
      const current = prev[field] ?? 1;
      const next = Math.max(1, Math.min(20, current + delta));
      return { ...prev, [field]: next };
    });
  };

  const handleNext = () => {
    if (currentStep === 1) {
      if (!formData.Area || formData.Area <= 0) {
        setValidationError("Please specify a valid usable area (> 0 m²).");
        return;
      }
    }
    setValidationError(null);
    if (currentStep < 4) {
      onStepChange(currentStep + 1);
    } else {
      onSubmit(formData);
    }
  };

  const handleBack = () => {
    setValidationError(null);
    if (currentStep > 1) {
      onStepChange(currentStep - 1);
    }
  };

  const currentArea = formData.Area ?? 78.7;
  const sliderArea = Math.min(500, Math.max(15, currentArea));

  const STEP_TITLES: Record<number, { title: string; desc: string }> = {
    1: { title: "Land Dimensions & Area", desc: "Specify usable floor area and lot boundaries" },
    2: { title: "Living Space & Architecture", desc: "Select room configuration and structural floors" },
    3: { title: "Location & Street Access", desc: "Select administrative district and plot position" },
    4: { title: "Property Attributes & Road", desc: "Choose property classification, road surface, and orientation" },
  };

  return (
    <div className="panel wizard-step-panel">
      {/* Step Header */}
      <div className="panel-header-row">
        <div>
          <span className="step-count-pill">STEP 0{currentStep} OF 04</span>
          <h2 className="panel-headline">{STEP_TITLES[currentStep]?.title}</h2>
          <p className="panel-subline">{STEP_TITLES[currentStep]?.desc}</p>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="btn-clear-link"
          disabled={loading}
        >
          Reset Inputs
        </button>
      </div>

      {/* Demo Presets Strip */}
      <div className="preset-strip">
        <span className="preset-strip-title">Quick Demo Presets:</span>
        <div className="preset-chip-list">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              className={`preset-chip ${activePresetId === preset.id ? "active" : ""}`}
              onClick={() => handleApplyPreset(preset)}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      {validationError && (
        <div className="validation-alert" role="alert">
          {validationError}
        </div>
      )}

      {/* STEP 1: Dimensions & Land */}
      {currentStep === 1 && (
        <div className="step-content-box animate-step">
          <div className="form-block">
            <div className="block-label-row">
              <label htmlFor="Area" className="block-label">
                Usable Area
              </label>
              <span className="area-value-badge">
                <strong>{currentArea.toLocaleString("en-US")}</strong> m²
              </span>
            </div>

            <div className="slider-wrapper">
              <input
                id="Area"
                type="range"
                min="15"
                max="500"
                step="1"
                value={sliderArea}
                onChange={(e) => {
                  setActivePresetId("");
                  onFormDataChange((prev) => ({ ...prev, Area: parseFloat(e.target.value) }));
                }}
                className="custom-range-slider"
              />
              <div className="slider-ticks">
                <span>15 m²</span>
                <span>250 m²</span>
                <span>500+ m²</span>
              </div>
            </div>
          </div>

          {/* Quick Area Size Cards */}
          <div className="cards-sub-section">
            <span className="cards-sub-label">Choose Common Area Benchmark:</span>
            <div className="option-cards-grid-4">
              {AREA_QUICK_CARDS.map((ac) => (
                <button
                  key={ac.area}
                  type="button"
                  className={`choice-card ${formData.Area === ac.area ? "selected" : ""}`}
                  onClick={() => {
                    setActivePresetId("");
                    onFormDataChange((prev) => ({ ...prev, Area: ac.area }));
                  }}
                >
                  <span className="card-opt-title">{ac.label}</span>
                  <span className="card-opt-desc">{ac.sub}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Lot Dimensions */}
          <div className="inputs-pair-grid">
            <div className="input-group">
              <label htmlFor="Width">Frontage Width (m)</label>
              <input
                id="Width"
                name="Width"
                type="number"
                step="0.1"
                min="0.5"
                max="500"
                value={formData.Width ?? ""}
                onChange={handleChange}
                placeholder="e.g. 4.0"
              />
            </div>

            <div className="input-group">
              <label htmlFor="Length">Lot Depth (m)</label>
              <input
                id="Length"
                name="Length"
                type="number"
                step="0.1"
                min="1"
                max="1000"
                value={formData.Length ?? ""}
                onChange={handleChange}
                placeholder="e.g. 19.6"
              />
            </div>
          </div>

          <div className="input-group" style={{ marginTop: "0.85rem" }}>
            <label htmlFor="Alley Width">Alley Access Width (m)</label>
            <input
              id="Alley Width"
              name="Alley Width"
              type="number"
              step="0.1"
              min="0"
              max="50"
              value={formData["Alley Width"] ?? ""}
              onChange={handleChange}
              placeholder="e.g. 3.5 (leave blank if frontage is on main road)"
            />
          </div>
        </div>
      )}

      {/* STEP 2: Living Space & Architecture */}
      {currentStep === 2 && (
        <div className="step-content-box animate-step">
          {/* Quick Layout Cards */}
          <div className="cards-sub-section">
            <span className="cards-sub-label">Choose Living Layout Architecture:</span>
            <div className="option-cards-grid-4">
              {LAYOUT_PRESET_CARDS.map((lp) => {
                const isSelected =
                  formData.Bedrooms === lp.beds &&
                  formData.Bathrooms === lp.baths &&
                  formData.Floors === lp.floors;
                return (
                  <button
                    key={lp.label}
                    type="button"
                    className={`choice-card ${isSelected ? "selected" : ""}`}
                    onClick={() => {
                      setActivePresetId("");
                      onFormDataChange((prev) => ({
                        ...prev,
                        Bedrooms: lp.beds,
                        Bathrooms: lp.baths,
                        Floors: lp.floors,
                      }));
                    }}
                  >
                    <span className="card-opt-title">{lp.label}</span>
                    <span className="card-opt-desc">
                      {lp.beds}B / {lp.baths}BA · {lp.floors}F
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Steppers */}
          <div className="steppers-grid">
            <div className="stepper-card">
              <label className="stepper-label">Bedrooms</label>
              <div className="stepper-control">
                <button
                  type="button"
                  className="stepper-btn"
                  onClick={() => updateCounter("Bedrooms", -1)}
                  aria-label="Decrease bedrooms"
                >
                  -
                </button>
                <span className="stepper-val">{formData.Bedrooms ?? 1}</span>
                <button
                  type="button"
                  className="stepper-btn"
                  onClick={() => updateCounter("Bedrooms", 1)}
                  aria-label="Increase bedrooms"
                >
                  +
                </button>
              </div>
            </div>

            <div className="stepper-card">
              <label className="stepper-label">Bathrooms</label>
              <div className="stepper-control">
                <button
                  type="button"
                  className="stepper-btn"
                  onClick={() => updateCounter("Bathrooms", -1)}
                  aria-label="Decrease bathrooms"
                >
                  -
                </button>
                <span className="stepper-val">{formData.Bathrooms ?? 1}</span>
                <button
                  type="button"
                  className="stepper-btn"
                  onClick={() => updateCounter("Bathrooms", 1)}
                  aria-label="Increase bathrooms"
                >
                  +
                </button>
              </div>
            </div>

            <div className="stepper-card">
              <label className="stepper-label">Total Floors</label>
              <div className="stepper-control">
                <button
                  type="button"
                  className="stepper-btn"
                  onClick={() => updateCounter("Floors", -1)}
                  aria-label="Decrease floors"
                >
                  -
                </button>
                <span className="stepper-val">{formData.Floors ?? 1}</span>
                <button
                  type="button"
                  className="stepper-btn"
                  onClick={() => updateCounter("Floors", 1)}
                  aria-label="Increase floors"
                >
                  +
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* STEP 3: Location & Position */}
      {currentStep === 3 && (
        <div className="step-content-box animate-step">
          <div className="cards-sub-section">
            <span className="cards-sub-label">Choose Street Access Position:</span>
            <div className="option-cards-grid-3">
              {POSITION_CARDS.map((pos) => (
                <button
                  key={pos.value}
                  type="button"
                  className={`choice-card ${formData.Position === pos.value ? "selected" : ""}`}
                  onClick={() => {
                    setActivePresetId("");
                    onFormDataChange((prev) => ({ ...prev, Position: pos.value }));
                  }}
                >
                  <span className="card-opt-title">{pos.title}</span>
                  <span className="card-opt-desc">{pos.desc}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="inputs-pair-grid">
            <div className="input-group">
              <label htmlFor="Province">Province / City</label>
              <select
                id="Province"
                name="Province"
                value={formData.Province ?? ""}
                onChange={handleChange}
              >
                {TOP_PROVINCES.map((p) => (
                  <option key={p.slug} value={p.slug}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="input-group">
              <label htmlFor="district">District / County</label>
              <input
                id="district"
                name="district"
                type="text"
                value={formData.district ?? ""}
                onChange={handleChange}
                placeholder="e.g. Rach Gia, District 7, Cau Giay"
              />
            </div>
          </div>

          <div className="input-group" style={{ marginTop: "0.85rem" }}>
            <label htmlFor="ward">Ward / Commune</label>
            <input
              id="ward"
              name="ward"
              type="text"
              value={formData.ward ?? ""}
              onChange={handleChange}
              placeholder="e.g. An Hoa Ward"
            />
          </div>
        </div>
      )}

      {/* STEP 4: Classification & Road Access */}
      {currentStep === 4 && (
        <div className="step-content-box animate-step">
          {/* Property Type Cards */}
          <div className="cards-sub-section">
            <span className="cards-sub-label">Choose Property Classification:</span>
            <div className="option-cards-grid-3">
              {PROPERTY_TYPE_CARDS.map((pt) => (
                <button
                  key={pt.value}
                  type="button"
                  className={`choice-card ${
                    formData["Property Type"] === pt.value ? "selected" : ""
                  }`}
                  onClick={() => {
                    setActivePresetId("");
                    onFormDataChange((prev) => ({ ...prev, "Property Type": pt.value }));
                  }}
                >
                  <span className="card-opt-title">{pt.title}</span>
                  <span className="card-opt-desc">{pt.desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Road Type Cards */}
          <div className="cards-sub-section">
            <span className="cards-sub-label">Choose Road Surface Access:</span>
            <div className="option-cards-grid-3">
              {ROAD_TYPE_CARDS.map((rt) => (
                <button
                  key={rt.value}
                  type="button"
                  className={`choice-card ${
                    formData["Road Type"] === rt.value ? "selected" : ""
                  }`}
                  onClick={() => {
                    setActivePresetId("");
                    onFormDataChange((prev) => ({ ...prev, "Road Type": rt.value }));
                  }}
                >
                  <span className="card-opt-title">{rt.title}</span>
                  <span className="card-opt-desc">{rt.desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Direction Cards */}
          <div className="cards-sub-section">
            <span className="cards-sub-label">Compass Direction:</span>
            <div className="option-cards-grid-4">
              {DIRECTION_CARDS.map((d) => (
                <button
                  key={d.value}
                  type="button"
                  className={`choice-card-sm ${formData.Direction === d.value ? "selected" : ""}`}
                  onClick={() => {
                    setActivePresetId("");
                    onFormDataChange((prev) => ({ ...prev, Direction: d.value }));
                  }}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          {/* Seller Role */}
          <div className="input-group" style={{ marginTop: "0.85rem" }}>
            <label htmlFor="Agent Role">Listed By</label>
            <select
              id="Agent Role"
              name="Agent Role"
              value={formData["Agent Role"] ?? "Chính chủ"}
              onChange={handleChange}
            >
              <option value="Chính chủ">Direct Owner (Chính chủ)</option>
              <option value="Môi giới">Real Estate Agent (Môi giới)</option>
            </select>
          </div>
        </div>
      )}

      {/* Step Navigation Actions */}
      <div className="wizard-actions-row">
        {currentStep > 1 && (
          <button
            type="button"
            className="btn-wizard-back"
            onClick={handleBack}
            disabled={loading}
          >
            ← Previous Step
          </button>
        )}

        <button
          type="button"
          className="btn-cyan-predict"
          onClick={handleNext}
          disabled={loading}
        >
          {loading
            ? "CALCULATING PREDICTION..."
            : currentStep < 4
            ? `PROCEED TO STEP 0${currentStep + 1} →`
            : "CALCULATE AI VALUATION"}
        </button>
      </div>
    </div>
  );
};
