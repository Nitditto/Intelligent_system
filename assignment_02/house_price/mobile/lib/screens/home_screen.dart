import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../core/theme/app_theme.dart';
import '../models/property_input.dart';
import '../providers/prediction_provider.dart';
import '../widgets/preset_bar.dart';
import '../widgets/result_card.dart';
import '../widgets/server_status_badge.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _formKey = GlobalKey<FormState>();

  // Text editing controllers
  late TextEditingController _areaCtrl;
  late TextEditingController _widthCtrl;
  late TextEditingController _lengthCtrl;
  late TextEditingController _alleyWidthCtrl;
  late TextEditingController _bedroomsCtrl;
  late TextEditingController _bathroomsCtrl;
  late TextEditingController _floorsCtrl;
  late TextEditingController _districtCtrl;
  late TextEditingController _wardCtrl;

  // Dropdown states
  String? _propertyType = 'Nhà riêng';
  String? _position = 'Đường chính';
  String? _direction = 'Nam';
  String? _roadType = 'Đường nhựa';
  String? _province = 'an-giang';
  String? _agentRole = 'Chính chủ';

  PropertyInput? _lastInput;

  final List<String> _propertyTypes = [
    'Nhà riêng',
    'Căn hộ chung cư',
    'Đất',
    'Nhà mặt phố',
    'Biệt thự',
    'Khác',
  ];

  final List<String> _positions = [
    'Đường chính',
    'Trong hẻm',
    'Mặt tiền',
  ];

  final List<String> _directions = [
    'Đông',
    'Tây',
    'Nam',
    'Bắc',
    'Đông Nam',
    'Đông Bắc',
    'Tây Nam',
    'Tây Bắc',
  ];

  final List<String> _roadTypes = [
    'Đường nhựa',
    'Đường bê tông',
    'Đường đất',
  ];

  final List<Map<String, String>> _provinces = [
    {'slug': 'an-giang', 'name': 'An Giang'},
    {'slug': 'tp-ho-chi-minh', 'name': 'Ho Chi Minh City'},
    {'slug': 'ha-noi', 'name': 'Hanoi'},
    {'slug': 'da-nang', 'name': 'Da Nang'},
    {'slug': 'binh-duong', 'name': 'Binh Duong'},
    {'slug': 'dong-nai', 'name': 'Dong Nai'},
    {'slug': 'can-tho', 'name': 'Can Tho'},
    {'slug': 'hai-phong', 'name': 'Hai Phong'},
    {'slug': 'khanh-hoa', 'name': 'Khanh Hoa'},
    {'slug': 'lam-dong', 'name': 'Lam Dong'},
  ];

  final List<String> _agentRoles = ['Chính chủ', 'Môi giới'];

  @override
  void initState() {
    super.initState();
    _areaCtrl = TextEditingController(text: '78.7');
    _widthCtrl = TextEditingController(text: '4.0');
    _lengthCtrl = TextEditingController(text: '19.6');
    _alleyWidthCtrl = TextEditingController(text: '3.5');
    _bedroomsCtrl = TextEditingController(text: '3');
    _bathroomsCtrl = TextEditingController(text: '2');
    _floorsCtrl = TextEditingController(text: '2');
    _districtCtrl = TextEditingController(text: 'Rạch Giá');
    _wardCtrl = TextEditingController(text: 'Phường An Hòa');
  }

  @override
  void dispose() {
    _areaCtrl.dispose();
    _widthCtrl.dispose();
    _lengthCtrl.dispose();
    _alleyWidthCtrl.dispose();
    _bedroomsCtrl.dispose();
    _bathroomsCtrl.dispose();
    _floorsCtrl.dispose();
    _districtCtrl.dispose();
    _wardCtrl.dispose();
    super.dispose();
  }

  void _applyPresetData(PropertyInput data) {
    setState(() {
      _areaCtrl.text = data.area.toString();
      _widthCtrl.text = data.width?.toString() ?? '';
      _lengthCtrl.text = data.length?.toString() ?? '';
      _alleyWidthCtrl.text = data.alleyWidth?.toString() ?? '';
      _bedroomsCtrl.text = data.bedrooms?.toString() ?? '';
      _bathroomsCtrl.text = data.bathrooms?.toString() ?? '';
      _floorsCtrl.text = data.floors?.toString() ?? '';
      _districtCtrl.text = data.district ?? '';
      _wardCtrl.text = data.ward ?? '';
      _propertyType = data.propertyType ?? _propertyType;
      _position = data.position ?? _position;
      _direction = data.direction ?? _direction;
      _roadType = data.roadType ?? _roadType;
      _province = data.province ?? _province;
      _agentRole = data.agentRole ?? _agentRole;
    });
  }

  void _handleSubmit() {
    if (!_formKey.currentState!.validate()) return;

    final input = PropertyInput(
      area: double.parse(_areaCtrl.text),
      width: double.tryParse(_widthCtrl.text),
      length: double.tryParse(_lengthCtrl.text),
      alleyWidth: double.tryParse(_alleyWidthCtrl.text),
      bedrooms: double.tryParse(_bedroomsCtrl.text),
      bathrooms: double.tryParse(_bathroomsCtrl.text),
      floors: double.tryParse(_floorsCtrl.text),
      district: _districtCtrl.text.isEmpty ? null : _districtCtrl.text,
      ward: _wardCtrl.text.isEmpty ? null : _wardCtrl.text,
      propertyType: _propertyType,
      position: _position,
      direction: _direction,
      roadType: _roadType,
      province: _province,
      agentRole: _agentRole,
    );

    setState(() {
      _lastInput = input;
    });

    context.read<PredictionProvider>().predict(input);
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<PredictionProvider>();

    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            Text('VN Real Estate Valuation', style: TextStyle(fontSize: 16)),
            Text('AI Automated Valuation Model (AVM)',
                style: TextStyle(fontSize: 11, color: AppTheme.textMuted)),
          ],
        ),
        actions: const [
          Padding(
            padding: EdgeInsets.only(right: 12.0),
            child: Center(child: ServerStatusBadge()),
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Preset bar
                PresetBar(onSelectPreset: _applyPresetData),
                const SizedBox(height: 16),

                // Error alert if any
                if (provider.errorMessage != null) ...[
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: const Color(0xFFFEF2F2),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: const Color(0xFFFECACA)),
                    ),
                    child: Text(
                      'Error: ${provider.errorMessage!}',
                      style: const TextStyle(color: Color(0xFFB91C1C), fontSize: 13),
                    ),
                  ),
                  const SizedBox(height: 16),
                ],

                // Result card if prediction available
                if (provider.result != null) ...[
                  ResultCard(
                    result: provider.result!,
                    input: _lastInput,
                    onClear: () => provider.clearResult(),
                  ),
                  const SizedBox(height: 20),
                ],

                // Section 1: Dimensions
                _buildSectionHeader('1. Land Dimensions & Usable Area'),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(14.0),
                    child: Column(
                      children: [
                        TextFormField(
                          controller: _areaCtrl,
                          keyboardType: const TextInputType.numberWithOptions(decimal: true),
                          decoration: const InputDecoration(
                            labelText: 'Usable Area (m²) *',
                            hintText: 'e.g. 78.7',
                          ),
                          validator: (val) {
                            if (val == null || val.isEmpty) return 'Usable area is required';
                            final n = double.tryParse(val);
                            if (n == null || n <= 0) return 'Area must be greater than 0 m²';
                            if (n > 10000) return 'Area cannot exceed 10,000 m²';
                            return null;
                          },
                        ),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            Expanded(
                              child: TextFormField(
                                controller: _widthCtrl,
                                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                                decoration: const InputDecoration(
                                  labelText: 'Frontage Width (m)',
                                  hintText: 'e.g. 4.0',
                                ),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: TextFormField(
                                controller: _lengthCtrl,
                                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                                decoration: const InputDecoration(
                                  labelText: 'Lot Depth (m)',
                                  hintText: 'e.g. 19.6',
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        TextFormField(
                          controller: _alleyWidthCtrl,
                          keyboardType: const TextInputType.numberWithOptions(decimal: true),
                          decoration: const InputDecoration(
                            labelText: 'Alley Width (m, if applicable)',
                            hintText: 'e.g. 3.5',
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),

                // Section 2: Rooms
                _buildSectionHeader('2. Living Spaces & Structure'),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(14.0),
                    child: Row(
                      children: [
                        Expanded(
                          child: TextFormField(
                            controller: _bedroomsCtrl,
                            keyboardType: TextInputType.number,
                            decoration: const InputDecoration(labelText: 'Bedrooms'),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: TextFormField(
                            controller: _bathroomsCtrl,
                            keyboardType: TextInputType.number,
                            decoration: const InputDecoration(labelText: 'Bathrooms'),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: TextFormField(
                            controller: _floorsCtrl,
                            keyboardType: TextInputType.number,
                            decoration: const InputDecoration(labelText: 'Floors'),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),

                // Section 3: Location
                _buildSectionHeader('3. Location & Geography'),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(14.0),
                    child: Column(
                      children: [
                        DropdownButtonFormField<String>(
                          initialValue: _province,
                          decoration: const InputDecoration(labelText: 'Province / City'),
                          items: _provinces.map((p) {
                            return DropdownMenuItem(value: p['slug'], child: Text(p['name']!));
                          }).toList(),
                          onChanged: (val) => setState(() => _province = val),
                        ),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            Expanded(
                              child: TextFormField(
                                controller: _districtCtrl,
                                decoration: const InputDecoration(
                                  labelText: 'District / County',
                                  hintText: 'e.g. Rach Gia, District 7',
                                ),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: TextFormField(
                                controller: _wardCtrl,
                                decoration: const InputDecoration(
                                  labelText: 'Ward / Commune',
                                  hintText: 'e.g. An Hoa Ward',
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        DropdownButtonFormField<String>(
                          initialValue: _position,
                          decoration: const InputDecoration(labelText: 'Property Position'),
                          items: _positions.map((pos) {
                            return DropdownMenuItem(value: pos, child: Text(pos));
                          }).toList(),
                          onChanged: (val) => setState(() => _position = val),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),

                // Section 4: Property Details
                _buildSectionHeader('4. Classification & Attributes'),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(14.0),
                    child: Column(
                      children: [
                        DropdownButtonFormField<String>(
                          initialValue: _propertyType,
                          decoration: const InputDecoration(labelText: 'Property Type'),
                          items: _propertyTypes.map((pt) {
                            return DropdownMenuItem(value: pt, child: Text(pt));
                          }).toList(),
                          onChanged: (val) => setState(() => _propertyType = val),
                        ),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            Expanded(
                              child: DropdownButtonFormField<String>(
                                initialValue: _direction,
                                decoration: const InputDecoration(labelText: 'Direction'),
                                items: _directions.map((d) {
                                  return DropdownMenuItem(value: d, child: Text(d));
                                }).toList(),
                                onChanged: (val) => setState(() => _direction = val),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: DropdownButtonFormField<String>(
                                initialValue: _roadType,
                                decoration: const InputDecoration(labelText: 'Road Access'),
                                items: _roadTypes.map((r) {
                                  return DropdownMenuItem(value: r, child: Text(r));
                                }).toList(),
                                onChanged: (val) => setState(() => _roadType = val),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        DropdownButtonFormField<String>(
                          initialValue: _agentRole,
                          decoration: const InputDecoration(labelText: 'Seller Role'),
                          items: _agentRoles.map((role) {
                            return DropdownMenuItem(value: role, child: Text(role));
                          }).toList(),
                          onChanged: (val) => setState(() => _agentRole = val),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Predict Button
                ElevatedButton(
                  onPressed: provider.isSubmitting ? null : _handleSubmit,
                  child: provider.isSubmitting
                      ? Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: const [
                            SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                            ),
                            SizedBox(width: 12),
                            Text('Calculating valuation via API...'),
                          ],
                        )
                      : const Text('Estimate Property Value'),
                ),
                const SizedBox(height: 24),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(left: 4, bottom: 8),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.w700,
          color: AppTheme.primaryColor,
        ),
      ),
    );
  }
}
