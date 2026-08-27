import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:vibration/vibration.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:geolocator/geolocator.dart';
import 'package:url_launcher/url_launcher.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Jacky TinyML Fall Monitor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // 💡 這裡的 UUID 必須與你的 V12 韌體 (nano33-fall-detection-v6-fixed-v12.ino) 保持絕對一致！
  final String targetDeviceName = "Jacky_FallMonitor";
  final String serviceUuid = "FFF0";
  final String stateCharUuid = "FFF1"; // 0:Standby, 1:Prefall, 2:Fall
  final String avmCharUuid = "FFF2";   // 實時 AVM 數據 (C++ float)

  BluetoothDevice? targetDevice;
  BluetoothCharacteristic? stateCharacteristic;
  BluetoothCharacteristic? avmCharacteristic;

  String connectionStateText = "未連線";
  int currentSystemState = 0; // 0: 正常, 1: 預警, 2: 跌倒
  double currentAvmValue = 9.806;
  bool isScanning = false;

  final AudioPlayer _audioPlayer = AudioPlayer();
  StreamSubscription<List<int>>? _stateSubscription;
  StreamSubscription<List<int>>? _avmSubscription;

  @override
  void initState() {
    super.initState();
    // 當藍牙硬體開啟時，自動開啟掃描
    FlutterBluePlus.adapterState.listen((state) {
      if (state == BluetoothAdapterState.on) {
        startScan();
      }
    });
  }

  // 1. 掃描藍牙外設，自動尋找你的 [Jacky_FallMonitor]
  void startScan() async {
    if (isScanning) return;
    setState(() {
      isScanning = true;
      connectionStateText = "正在搜尋手環...";
    });

    await FlutterBluePlus.startScan(timeout: const Duration(seconds: 15));

    FlutterBluePlus.scanResults.listen((results) async {
      for (ScanResult r in results) {
        if (r.device.localName == targetDeviceName) {
          FlutterBluePlus.stopScan();
          setState(() {
            targetDevice = r.device;
            isScanning = false;
            connectionStateText = "已找到手環，正在連線...";
          });
          connectToDevice();
          break;
        }
      }
    });
  }

  // 2. 建立連線並自動開啟 CCCD 監聽 (Notify)
  void connectToDevice() async {
    if (targetDevice == null) return;

    try {
      await targetDevice!.connect(autoConnect: true);
      setState(() {
        connectionStateText = "手環連線成功 ✅";
      });

      // 監聽斷線事件，實現斷線自動重新掃描連線
      targetDevice!.connectionState.listen((state) {
        if (state == BluetoothConnectionState.disconnected) {
          setState(() {
            connectionStateText = "連線中斷 ❌";
            currentSystemState = 0;
            currentAvmValue = 9.806;
          });
          _audioPlayer.stop();
          Vibration.cancel();
          startScan(); // 自動重新搜集
        }
      });

      // 搜尋手環內置的 FFF0 遙測服務
      List<BluetoothService> services = await targetDevice!.discoverServices();
      for (BluetoothService service in services) {
        if (service.uuid.toString().toUpperCase().contains(serviceUuid)) {
          for (BluetoothCharacteristic char in service.characteristics) {
            
            // 訂閱 FFF1: 跌倒狀態 (0/1/2)
            if (char.uuid.toString().toUpperCase().contains(stateCharUuid)) {
              stateCharacteristic = char;
              await char.setNotifyValue(true); // 💡 向底層 CCCD 寫入 0x0001 開啟訂閱！
              _stateSubscription = char.onValueReceived.listen((value) {
                if (value.isNotEmpty) {
                  int receivedState = value[0]; // 讀取狀態 byte
                  setState(() {
                    currentSystemState = receivedState;
                  });
                  handleStateAction(receivedState); // 執行手機端物理反應
                }
              });
            }

            // 訂閱 FFF2: 實時 AVM 物理波形
            if (char.uuid.toString().toUpperCase().contains(avmCharUuid)) {
              avmCharacteristic = char;
              await char.setNotifyValue(true);
              _avmSubscription = char.onValueReceived.listen((value) {
                if (value.isNotEmpty) {
                  // 💡 解析 V12 韌體傳來的 4-byte IEEE 754 浮點數
                  double parsedAvm = _parseBluetoothFloat(value);
                  setState(() {
                    currentAvmValue = parsedAvm;
                  });
                }
              });
            }
          }
        }
      }
    } catch (e) {
      setState(() {
        connectionStateText = "連線失敗: $e";
      });
    }
  }

  // 💡 核心轉換：將單片機 C++ 傳來的 4位元組(Float) 數據還原為 Dart 的 double 類型
  double _parseBluetoothFloat(List<int> bytes) {
    if (bytes.length < 4) return 9.806;
    final Uint8List list = Uint8List.fromList(bytes);
    final ByteData byteData = ByteData.sublistView(list);
    return byteData.getFloat32(0, Endian.little); // V12.ino 發送的是 little-endian 格式
  }

  // 3. 核心決策與手機物理反應分流
  void handleStateAction(int state) async {
    if (state == 0) {
      // 正常狀態：停止所有震動與警報音
      _audioPlayer.stop(); 
      Vibration.cancel();
    } 
    else if (state == 1) {
      // ⚠️ Prefall 預警：手機輕微震動 0.5 秒，提醒照護者注意
      if (await Vibration.hasVibrator() ?? false) {
        Vibration.vibrate(duration: 500, amplitude: 128); 
      }
    } 
    else if (state == 2) {
      // 🚨 CONFIRMED FALL: 嚴重跌倒！
      // A. 觸發強烈、持續不間斷的交替震動 (震動 500ms, 暫停 200ms)
      if (await Vibration.hasVibrator() ?? false) {
        Vibration.vibrate(pattern: [500, 200, 500, 200], repeat: 0); 
      }

      // B. 播放高分貝警笛聲 (此處使用 SoundJay 提供的公開預設警笛音訊進行測試)
      try {
        await _audioPlayer.play(UrlSource('https://www.soundjay.com/buttons/beep-01a.mp3')); 
      } catch (e) {
        print("音效播放失敗: $e");
      }

      // C. 獲取高精度 GPS 定位，並調用系統簡訊界面發送 SOS
      sendSmsAlert();
    }
  }

  // 4. 高精度 GPS 定位獲取與自動喚醒簡訊
  void sendSmsAlert() async {
    bool serviceEnabled;
    LocationPermission permission;

    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return;

    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) return;
    }

    // 獲取目前最高精度的 GPS 位置
    Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high);

    // 生成 Google Maps 地圖定位求救連結
    String mapUrl = "https://www.google.com/maps/search/?api=1&query=${position.latitude},${position.longitude}";
    String sosMessage = "【緊急求救】Jacky 跌倒監測系統檢測到使用者發生嚴重跌倒！目前位置：$mapUrl  請立刻前往協助！";

    // 配置照護者的電話號碼 (發送簡訊)
    final Uri smsUri = Uri(
      scheme: 'sms',
      path: '+85212345678', // 💡 在這裡修改為你用來測試的看護者真實手機號碼
      queryParameters: <String, String>{
        'body': sosMessage,
      },
    );

    if (await launchUrl(smsUri)) {
      print("SOS 簡訊界面已成功拉起");
    } else {
      print("無法拉起系統簡訊界面");
    }
  }

  // 5. 高對比、醫療監護級 App 介面
  @override
  Widget build(BuildContext context) {
    Color statusColor = Colors.green;
    String statusText = "安全監護中";
    IconData statusIcon = Icons.security;

    if (currentSystemState == 1) {
      statusColor = Colors.orange;
      statusText = "跌倒前兆預警 (Prefall)！";
      statusIcon = Icons.warning_amber_rounded;
    } else if (currentSystemState == 2) {
      statusColor = Colors.red;
      statusText = "嚴重跌倒 (Fall Crash)！！";
      statusIcon = Icons.gavel_rounded;
    }

    return Scaffold(
      backgroundColor: const Color(0xFFF5F5F7),
      appBar: AppBar(
        title: const Text("Jacky's Active Fall Monitor", style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.white,
        centerTitle: true,
        elevation: 1,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // 系統連線狀態
            Text("通訊狀態: $connectionStateText", 
              style: const TextStyle(fontSize: 16, color: Colors.grey, fontWeight: FontWeight.w500)),
            const SizedBox(height: 35),

            // 狀態指示大圓盤 (高視覺張力)
            AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              width: 240,
              height: 240,
              decoration: BoxDecoration(
                color: statusColor,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: statusColor.withOpacity(0.4),
                    blurRadius: 25,
                    spreadRadius: 8,
                  )
                ],
              ),
              child: Icon(statusIcon, size: 110, color: Colors.white),
            ),
            const SizedBox(height: 45),

            // 狀態文本
            Text(statusText, 
              style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold, color: statusColor)),
            const SizedBox(height: 12),

            // 實時 AVM 運動強度顯示
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(15),
                boxShadow: [
                  BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 10)
                ],
              ),
              child: Text("實時運動指標 AVM: ${currentAvmValue.toStringAsFixed(3)} m/s²", 
                style: const TextStyle(fontSize: 16, fontFamily: 'monospace', fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 45),

            // 手動重新搜尋按鈕
            ElevatedButton.icon(
              onPressed: startScan,
              icon: const Icon(Icons.bluetooth_searching),
              label: const Text("手動搜尋手環"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 35, vertical: 15),
                textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _stateSubscription?.cancel();
    _avmSubscription?.cancel();
    _audioPlayer.dispose();
    super.dispose();
  }
}