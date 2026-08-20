import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Building, MapPin, Ruler, Compass, Search, Home, ChevronRight, CheckCircle2 } from 'lucide-react';
import localProvinces from './locations.json';
import { ModelDetailsPanel } from './components/ModelDetailsPanel';
import { Browser } from '@capacitor/browser';

const API_BASE = 'http://localhost:8001/api';

function App() {
  const [locations, setLocations] = useState<Record<string, string[]>>({});
  const [cities, setCities] = useState<string[]>([]);

  const [form, setForm] = useState({
    StreetAddress: '',
    City: 'Thành phố Hà Nội',
    District: 'Quận Hai Bà Trưng',
    Area: 50,
    Frontage: 4,
    Access_Road: 3,
    House_direction: 'Đông - Nam',
    Balcony_direction: 'Đông - Nam',
    Floors: 3,
    Bedrooms: 3,
    Bathrooms: 2,
    Legal_status: 'Sổ đỏ/ Sổ hồng',
    Furniture_state: 'Cơ bản'
  });

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [resultId, setResultId] = useState<string | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  useEffect(() => {
    // Process localProvinces JSON
    const locMap: Record<string, string[]> = {};
    const cityList: string[] = [];
    localProvinces.forEach((prov: any) => {
      cityList.push(prov.name);
      locMap[prov.name] = prov.districts.map((d: any) => d.name);
    });

    setLocations(locMap);
    setCities(cityList);
    if (cityList.length > 0) {
      const firstCity = cityList[0];
      setForm(f => ({ ...f, City: firstCity, District: locMap[firstCity]?.[0] || '' }));
    }
  }, []);

  const handleCityChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const city = e.target.value;
    setForm(f => ({
      ...f,
      City: city,
      District: locations[city]?.[0] || ''
    }));
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setForm(f => ({
      ...f,
      [name]: type === 'number' ? parseFloat(value) || 0 : value
    }));
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResults([]);
    setSearchResults([]);
    try {
      const res = await axios.post(`${API_BASE}/predict`, form);
      // Sort so XGBoost or best is first
      res.data.results.sort((a: any, b: any) => {
        // Just hardcode a preference for tree models or just return them
        return b.price - a.price; // sort by price for now, or backend already sorted?
      });
      setResults(res.data.results);
      setResultId(res.data.id);

      // Auto trigger search for the first model's price
      if (res.data.results.length > 0) {
        const bestPrice = res.data.results.find((r: any) => r.model === 'xgboost')?.price || res.data.results[0].price;
        fetchRealHouses(bestPrice);
      }
    } catch (err) {
      console.error(err);
      alert("Lỗi khi dự đoán.");
    } finally {
      setLoading(false);
    }
  };

  const fetchRealHouses = async (price: number) => {
    setSearchLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/search`, {
        params: {
          city: form.City,
          district: form.District,
          price: price,
          area: form.Area,
          floors: form.Floors,
          bedrooms: form.Bedrooms
        }
      });
      setSearchResults(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setSearchLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 pb-20">
      <div className="bg-gradient-to-r from-brand-600 to-brand-900 pb-32 pt-12">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-4xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <Home className="w-10 h-10" />
            AI Định Giá Bất Động Sản
          </h1>
          <p className="mt-4 text-brand-100 max-w-2xl text-lg">
            Sử dụng 6 mô hình Machine Learning tiên tiến để dự đoán giá nhà chính xác và tìm kiếm các bất động sản tương tự trên thị trường.
          </p>
        </div>
      </div>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 -mt-24">
        <div className="glass-card p-6 sm:p-8 mb-8">
          <form onSubmit={handlePredict} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

              {/* Location */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2 border-b pb-2">
                  <MapPin className="w-5 h-5 text-brand-500" /> Khu vực
                </h3>
                <div className="grid grid-cols-1 gap-4">
                  <div>
                    <label className="label-text">Địa chỉ (Số nhà, đường...)</label>
                    <input type="text" name="StreetAddress" value={form.StreetAddress} onChange={handleChange} className="input-field" placeholder="Ví dụ: 123 Đường Nguyễn Văn Linh" />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div>
                    <label className="label-text">Tỉnh/Thành phố</label>
                    <select name="City" value={form.City} onChange={handleCityChange} className="input-field">
                      {cities.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="label-text">Quận/Huyện</label>
                    <select name="District" value={form.District} onChange={handleChange} className="input-field">
                      {locations[form.City]?.map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                  </div>
                </div>
              </div>

              {/* Specs */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2 border-b pb-2">
                  <Ruler className="w-5 h-5 text-brand-500" /> Kích thước & Số phòng
                </h3>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="label-text">Diện tích (m²)</label>
                    <input type="number" name="Area" value={form.Area} onChange={handleChange} className="input-field" step="0.1" />
                  </div>
                  <div>
                    <label className="label-text">Mặt tiền (m)</label>
                    <input type="number" name="Frontage" value={form.Frontage} onChange={handleChange} className="input-field" step="0.1" />
                  </div>
                  <div>
                    <label className="label-text">Ngõ vào (m)</label>
                    <input type="number" name="Access_Road" value={form.Access_Road} onChange={handleChange} className="input-field" step="0.1" />
                  </div>
                  <div>
                    <label className="label-text">Số tầng</label>
                    <input type="number" name="Floors" value={form.Floors} onChange={handleChange} className="input-field" />
                  </div>
                  <div>
                    <label className="label-text">Phòng ngủ</label>
                    <input type="number" name="Bedrooms" value={form.Bedrooms} onChange={handleChange} className="input-field" />
                  </div>
                  <div>
                    <label className="label-text">Phòng tắm</label>
                    <input type="number" name="Bathrooms" value={form.Bathrooms} onChange={handleChange} className="input-field" />
                  </div>
                </div>
              </div>

              {/* Other Specs */}
              <div className="space-y-4 md:col-span-2">
                <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2 border-b pb-2">
                  <Compass className="w-5 h-5 text-brand-500" /> Thông tin khác
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <label className="label-text">Hướng nhà</label>
                    <select name="House_direction" value={form.House_direction} onChange={handleChange} className="input-field">
                      <option value="Đông">Đông</option>
                      <option value="Tây">Tây</option>
                      <option value="Nam">Nam</option>
                      <option value="Bắc">Bắc</option>
                      <option value="Đông - Nam">Đông - Nam</option>
                      <option value="Tây - Nam">Tây - Nam</option>
                      <option value="Đông - Bắc">Đông - Bắc</option>
                      <option value="Tây - Bắc">Tây - Bắc</option>
                    </select>
                  </div>
                  <div>
                    <label className="label-text">Hướng ban công</label>
                    <select name="Balcony_direction" value={form.Balcony_direction} onChange={handleChange} className="input-field">
                      <option value="Đông">Đông</option>
                      <option value="Tây">Tây</option>
                      <option value="Nam">Nam</option>
                      <option value="Bắc">Bắc</option>
                      <option value="Đông - Nam">Đông - Nam</option>
                      <option value="Tây - Nam">Tây - Nam</option>
                      <option value="Đông - Bắc">Đông - Bắc</option>
                      <option value="Tây - Bắc">Tây - Bắc</option>
                    </select>
                  </div>
                  <div>
                    <label className="label-text">Pháp lý</label>
                    <select name="Legal_status" value={form.Legal_status} onChange={handleChange} className="input-field">
                      <option value="Have certificate">Đã có sổ đỏ/sổ hồng</option>
                      <option value="Waiting for certificate">Đang chờ sổ</option>
                      <option value="Sale contract">Hợp đồng mua bán</option>
                    </select>
                  </div>
                  <div>
                    <label className="label-text">Nội thất</label>
                    <select name="Furniture_state" value={form.Furniture_state} onChange={handleChange} className="input-field">
                      <option value="Full">Đầy đủ</option>
                      <option value="Basic">Cơ bản</option>
                      <option value="None">Không có</option>
                    </select>
                  </div>
                </div>
              </div>

            </div>

            <div className="pt-4">
              <button
                type="submit"
                disabled={loading}
                className="w-full md:w-auto px-8 py-4 bg-brand-600 hover:bg-brand-700 text-white rounded-xl font-bold text-lg shadow-lg hover:shadow-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {loading ? <span className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></span> : <Building className="w-6 h-6" />}
                Dự đoán giá trị
              </button>
            </div>
          </form>
        </div>

        {/* Results */}
        {results.length > 0 && (
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-slate-800 mb-6 flex items-center gap-2">
              <CheckCircle2 className="w-6 h-6 text-green-500" />
              Kết quả dự đoán từ 6 mô hình
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {results.map((r, i) => (
                <div
                  key={i}
                  onClick={() => {
                    setSelectedModel(r.model);
                    setIsPanelOpen(true);
                  }}
                  className={`p-4 rounded-xl border-2 transition-all cursor-pointer hover:shadow-lg hover:-translate-y-1 transform group ${r.model === 'xgboost' || r.model === 'best_model' ? 'bg-green-50 border-green-500 shadow-md scale-105 relative z-10' : 'bg-white border-slate-200'}`}
                >
                  {(r.model === 'xgboost' || r.model === 'best_model') && <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-green-500 text-white text-xs font-bold px-2 py-1 rounded-full">TỐT NHẤT</div>}
                  <div className="text-sm text-slate-500 uppercase tracking-wider font-semibold mb-1">{r.model.replace('_', ' ')}</div>
                  <div className={`text-2xl font-bold ${r.model === 'xgboost' || r.model === 'best_model' ? 'text-green-700' : 'text-slate-800'}`}>
                    {r.price.toFixed(2)} tỷ
                  </div>
                  <div className="text-xs text-slate-400 mt-1.5 flex flex-col gap-0.5">
                    <div>Độ tin cậy: {(r.confidence * 100).toFixed(1)}%</div>
                    <div className="text-[10px] text-slate-500 flex justify-between mt-1 pt-1 border-t border-slate-100/50">
                      <span>R²: {r.r2.toFixed(4)}</span>
                      <span>MAPE: {r.mape.toFixed(1)}%</span>
                    </div>
                  </div>

                  <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-brand-600 font-medium group-hover:text-brand-700">
                    <span>Xem đóng góp SHAP</span>
                    <ChevronRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Real Houses Search */}
        {(searchLoading || results.length > 0) && (
          <div className="mt-8">
            <h2 className="text-2xl font-bold text-slate-800 mb-6 flex items-center gap-2">
              <Search className="w-6 h-6 text-brand-500" />
              Nhà đất thực tế đang bán
            </h2>
            {searchLoading ? (
              <div className="flex justify-center p-12">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-brand-500 border-t-transparent"></div>
              </div>
            ) : searchResults.length === 0 ? (
              <div className="bg-slate-50 border border-slate-200 rounded-2xl p-12 text-center">
                <div className="w-16 h-16 bg-slate-200 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Search className="w-8 h-8 text-slate-400" />
                </div>
                <h3 className="text-lg font-bold text-slate-700 mb-2">Không tìm thấy bất động sản phù hợp</h3>
                <p className="text-slate-500">Hệ thống đã thử tìm trên các trang batdongsan.com.vn, bds68 nhưng hiện không có tin đăng nào khớp hoàn toàn với vị trí và mức giá dự đoán. Vui lòng thử lại với diện tích hoặc vị trí khác.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {searchResults.map((item, idx) => (
                  <a key={idx} href={item.link} onClick={(e) => { e.preventDefault(); Browser.open({ url: item.link }).catch(() => window.open(item.link, '_blank')); }} className="group glass-card overflow-hidden hover:shadow-2xl transition-all flex flex-col">
                    <div className="h-48 overflow-hidden bg-slate-200 relative">
                      {item.image && item.image !== "No Images" ? (
                        <img
                          src={item.image}
                          alt="House"
                          className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                          onError={(e) => {
                            e.currentTarget.style.display = 'none';
                            if (e.currentTarget.nextElementSibling) {
                              (e.currentTarget.nextElementSibling as HTMLElement).style.display = 'flex';
                            }
                          }}
                        />
                      ) : null}
                      <div className="absolute inset-0 bg-slate-200 text-slate-500 flex items-center justify-center font-medium" style={{ display: (!item.image || item.image === "No Images") ? 'flex' : 'none' }}>
                        No Images
                      </div>
                      <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm z-10">Xem chi tiết</div>
                    </div>
                    <div className="p-5 flex-1 flex flex-col">
                      <h3 className="font-semibold text-slate-800 line-clamp-2 mb-2 group-hover:text-brand-600 transition-colors">{item.title}</h3>
                      <p className="text-sm text-slate-500 line-clamp-3 mb-4 flex-1">{item.snippet}</p>
                      <div className="mt-auto flex items-center text-brand-600 font-medium text-sm">
                        Truy cập trang <ChevronRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
                      </div>
                    </div>
                  </a>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      <ModelDetailsPanel
        isOpen={isPanelOpen}
        onClose={() => setIsPanelOpen(false)}
        model={selectedModel}
        resultId={resultId}
        apiUrl={API_BASE.replace('/api', '')}
      />
    </div>
  );
}

export default App;
