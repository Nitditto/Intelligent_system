import React, { useState } from "react";
import { PRESETS, type Preset } from "../constants/options";
import type { ListingInput } from "../types";

interface PredictionFormProps {
  onSubmit: (data: ListingInput) => void;
  loading: boolean;
  onReset: () => void;
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

const PROPERTY_TYPES = [
  "Nhà riêng",
  "Căn hộ chung cư",
  "Đất",
  "Nhà mặt phố",
  "Biệt thự",
  "Nhà trọ, phòng trọ",
  "Kho, nhà xưởng",
  "Bất động sản khác",
];

const POSITIONS = ["Đường chính", "Trong hẻm", "Mặt tiền"];
const DIRECTIONS = ["Đông", "Tây", "Nam", "Bắc", "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc"];
const ROAD_TYPES = ["Đường nhựa", "Đường bê tông", "Đường đất"];
const AGENT_ROLES = ["Chính chủ", "Môi giới"];

export const PredictionForm: React.FC<PredictionFormProps> = ({
  onSubmit,
  loading,
  onReset,
}) => {
  const [formData, setFormData] = useState<ListingInput>({
    Area: 78.7,
    Width: 4.0,
    Length: 19.6,
    Bedrooms: 3,
    Bathrooms: 2,
    Floors: 2,
    "Alley Width": 3.5,
    "Agent Listing Count": 1,
    "Property Type": "Nhà riêng",
    Position: "Đường chính",
    Direction: "Nam",
    "Road Type": "Đường nhựa",
    Province: "an-giang",
    "Agent Role": "Chính chủ",
    ward: "Phường An Hòa",
    district: "Rạch Giá",
  });

  const [activePresetId, setActivePresetId] = useState<string>("rach-gia-full");
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleApplyPreset = (preset: Preset) => {
    setActivePresetId(preset.id);
    setFormData(preset.data);
    setValidationError(null);
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
    setActivePresetId("");

    if (type === "number") {
      const parsed = value === "" ? undefined : parseFloat(value);
      setFormData((prev) => ({ ...prev, [name]: parsed }));
    } else {
      setFormData((prev) => ({ ...prev, [name]: value === "" ? undefined : value }));
    }
  };

  const setNumberField = (name: keyof ListingInput, val: number) => {
    setActivePresetId("");
    setFormData((prev) => ({ ...prev, [name]: val }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.Area || formData.Area <= 0) {
      setValidationError("Usable Area is required and must be greater than 0 m².");
      return;
    }

    if (formData.Area > 10000) {
      setValidationError("Area cannot exceed 10,000 m².");
      return;
    }

    setValidationError(null);
    onSubmit(formData);
  };

  const handleClear = () => {
    setFormData({ Area: 70 });
    setActivePresetId("");
    setValidationError(null);
    onReset();
  };

  // UX Helper: calculate land footprint if width & length are provided
  const footprint =
    formData.Width && formData.Length
      ? (formData.Width * formData.Length).toFixed(1)
      : null;

  return (
    <div className="card form-card">
      <div className="card-header">
        <div>
          <h2 className="card-title">Property Inputs</h2>
          <p className="card-desc">Enter attributes to calculate the estimated market price.</p>
        </div>
        <button
          type="button"
          onClick={handleClear}
          className="link-button"
          disabled={loading}
        >
          Reset
        </button>
      </div>

      {/* Preset bar */}
      <div className="preset-container">
        <span className="preset-label">Quick Presets:</span>
        <div className="preset-pills">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              className={`preset-pill ${activePresetId === preset.id ? "active" : ""}`}
              onClick={() => handleApplyPreset(preset)}
              title={preset.description}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      {validationError && (
        <div className="error-alert" role="alert">
          {validationError}
        </div>
      )}

      <form onSubmit={handleSubmit} noValidate>
        {/* Section 1: Dimensions */}
        <fieldset className="form-group">
          <legend className="group-title">1. Dimensions & Land</legend>
          
          <div className="field-grid-2">
            <div className="input-block">
              <label htmlFor="Area">
                Usable Area <span className="req">*</span>
              </label>
              <div className="input-affix-wrapper">
                <input
                  id="Area"
                  name="Area"
                  type="number"
                  step="0.1"
                  min="1"
                  max="10000"
                  value={formData.Area ?? ""}
                  onChange={handleChange}
                  placeholder="78.7"
                  required
                />
                <span className="affix">m²</span>
              </div>
            </div>

            <div className="input-block">
              <label htmlFor="Width">Frontage Width</label>
              <div className="input-affix-wrapper">
                <input
                  id="Width"
                  name="Width"
                  type="number"
                  step="0.1"
                  min="0.5"
                  max="500"
                  value={formData.Width ?? ""}
                  onChange={handleChange}
                  placeholder="4.0"
                />
                <span className="affix">m</span>
              </div>
            </div>

            <div className="input-block">
              <label htmlFor="Length">Lot Depth</label>
              <div className="input-affix-wrapper">
                <input
                  id="Length"
                  name="Length"
                  type="number"
                  step="0.1"
                  min="1"
                  max="1000"
                  value={formData.Length ?? ""}
                  onChange={handleChange}
                  placeholder="19.6"
                />
                <span className="affix">m</span>
              </div>
            </div>

            <div className="input-block">
              <label htmlFor="Alley Width">Alley Access Width</label>
              <div className="input-affix-wrapper">
                <input
                  id="Alley Width"
                  name="Alley Width"
                  type="number"
                  step="0.1"
                  min="0"
                  max="50"
                  value={formData["Alley Width"] ?? ""}
                  onChange={handleChange}
                  placeholder="3.5 (if in alley)"
                />
                <span className="affix">m</span>
              </div>
            </div>
          </div>

          {footprint && (
            <div className="inline-hint">
              Ground footprint: <strong>{footprint} m²</strong> ({formData.Width}m × {formData.Length}m)
            </div>
          )}
        </fieldset>

        {/* Section 2: Rooms & Floors with intuitive tap chips */}
        <fieldset className="form-group">
          <legend className="group-title">2. Space & Structure</legend>
          
          <div className="counter-row">
            <div className="counter-block">
              <label>Bedrooms</label>
              <div className="segmented-selector">
                {[1, 2, 3, 4, 5].map((num) => (
                  <button
                    key={num}
                    type="button"
                    className={`segment-btn ${formData.Bedrooms === num ? "selected" : ""}`}
                    onClick={() => setNumberField("Bedrooms", num)}
                  >
                    {num === 5 ? "5+" : num}
                  </button>
                ))}
              </div>
            </div>

            <div className="counter-block">
              <label>Bathrooms</label>
              <div className="segmented-selector">
                {[1, 2, 3, 4].map((num) => (
                  <button
                    key={num}
                    type="button"
                    className={`segment-btn ${formData.Bathrooms === num ? "selected" : ""}`}
                    onClick={() => setNumberField("Bathrooms", num)}
                  >
                    {num === 4 ? "4+" : num}
                  </button>
                ))}
              </div>
            </div>

            <div className="counter-block">
              <label>Total Floors</label>
              <div className="segmented-selector">
                {[1, 2, 3, 4, 5].map((num) => (
                  <button
                    key={num}
                    type="button"
                    className={`segment-btn ${formData.Floors === num ? "selected" : ""}`}
                    onClick={() => setNumberField("Floors", num)}
                  >
                    {num === 5 ? "5+" : num}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </fieldset>

        {/* Section 3: Location */}
        <fieldset className="form-group">
          <legend className="group-title">3. Location</legend>
          
          <div className="field-grid-2">
            <div className="input-block">
              <label htmlFor="Province">Province / City</label>
              <select
                id="Province"
                name="Province"
                value={formData.Province ?? ""}
                onChange={handleChange}
              >
                <option value="">-- Select Province --</option>
                {TOP_PROVINCES.map((p) => (
                  <option key={p.slug} value={p.slug}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="input-block">
              <label htmlFor="district">District</label>
              <input
                id="district"
                name="district"
                type="text"
                value={formData.district ?? ""}
                onChange={handleChange}
                placeholder="e.g. Rach Gia, District 7"
              />
            </div>

            <div className="input-block">
              <label htmlFor="ward">Ward</label>
              <input
                id="ward"
                name="ward"
                type="text"
                value={formData.ward ?? ""}
                onChange={handleChange}
                placeholder="e.g. An Hoa"
              />
            </div>

            <div className="input-block">
              <label htmlFor="Position">Plot Position</label>
              <select
                id="Position"
                name="Position"
                value={formData.Position ?? ""}
                onChange={handleChange}
              >
                <option value="">-- Select Position --</option>
                {POSITIONS.map((pos) => (
                  <option key={pos} value={pos}>
                    {pos}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </fieldset>

        {/* Section 4: Property Details */}
        <fieldset className="form-group">
          <legend className="group-title">4. Classification & Road</legend>
          
          <div className="field-grid-2">
            <div className="input-block">
              <label htmlFor="Property Type">Property Type</label>
              <select
                id="Property Type"
                name="Property Type"
                value={formData["Property Type"] ?? ""}
                onChange={handleChange}
              >
                <option value="">-- Select Property Type --</option>
                {PROPERTY_TYPES.map((pt) => (
                  <option key={pt} value={pt}>
                    {pt}
                  </option>
                ))}
              </select>
            </div>

            <div className="input-block">
              <label htmlFor="Direction">Compass Direction</label>
              <select
                id="Direction"
                name="Direction"
                value={formData.Direction ?? ""}
                onChange={handleChange}
              >
                <option value="">-- Select Direction --</option>
                {DIRECTIONS.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>

            <div className="input-block">
              <label htmlFor="Road Type">Road Access</label>
              <select
                id="Road Type"
                name="Road Type"
                value={formData["Road Type"] ?? ""}
                onChange={handleChange}
              >
                <option value="">-- Select Road Type --</option>
                {ROAD_TYPES.map((rt) => (
                  <option key={rt} value={rt}>
                    {rt}
                  </option>
                ))}
              </select>
            </div>

            <div className="input-block">
              <label htmlFor="Agent Role">Listed By</label>
              <select
                id="Agent Role"
                name="Agent Role"
                value={formData["Agent Role"] ?? ""}
                onChange={handleChange}
              >
                <option value="">-- Select Role --</option>
                {AGENT_ROLES.map((ar) => (
                  <option key={ar} value={ar}>
                    {ar}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </fieldset>

        {/* Primary Action Button */}
        <div className="submit-wrapper">
          <button
            type="submit"
            className="btn-calculate"
            disabled={loading}
          >
            {loading ? "Calculating Estimate..." : "Calculate Price Estimate"}
          </button>
          <span className="submit-hint">Press Enter or click to estimate</span>
        </div>
      </form>
    </div>
  );
};
