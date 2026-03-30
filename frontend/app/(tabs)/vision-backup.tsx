import { ScrollView, Text, View, Pressable } from "react-native";
import { useState } from "react";
import { ScreenContainer } from "@/components/screen-container";
import { useColors } from "@/hooks/use-colors";
import * as Haptics from "expo-haptics";

interface OCRData {
  label: string;
  value: string;
  confidence: number;
}

interface GPUMetrics {
  utilization: number;
  temperature: number;
  memory: number;
}

const OCR_RESULTS: OCRData[] = [
  { label: "Runs", value: "156", confidence: 0.98 },
  { label: "Wickets", value: "3", confidence: 0.96 },
  { label: "Overs", value: "18.3", confidence: 0.94 },
  { label: "Target", value: "165", confidence: 0.92 },
];

const GPU_METRICS: GPUMetrics = {
  utilization: 62,
  temperature: 58,
  memory: 4200,
};

function ConfidenceBar({ confidence }: { confidence: number }) {
  const colors = useColors();
  const percentage = confidence * 100;

  return (
    <View className="flex-row items-center gap-2">
      <View className="flex-1 h-2 bg-background rounded-full overflow-hidden">
        <View
          className="h-full bg-success rounded-full"
          style={{ width: `${percentage}%` }}
        />
      </View>
      <Text className="text-xs font-semibold text-foreground w-12">
        {percentage.toFixed(0)}%
      </Text>
    </View>
  );
}

function OCRCard({ data }: { data: OCRData }) {
  return (
    <View className="bg-surface rounded-lg p-4 mb-3 border border-border">
      <View className="flex-row justify-between items-center mb-2">
        <Text className="text-sm font-semibold text-foreground">{data.label}</Text>
        <Text className="text-lg font-bold text-primary">{data.value}</Text>
      </View>
      <ConfidenceBar confidence={data.confidence} />
    </View>
  );
}

function MetricCard({
  label,
  value,
  unit,
  status,
}: {
  label: string;
  value: number;
  unit: string;
  status: "good" | "warning" | "critical";
}) {
  const statusColors = {
    good: "bg-success/10 border-success",
    warning: "bg-warning/10 border-warning",
    critical: "bg-error/10 border-error",
  };

  return (
    <View className={`${statusColors[status]} rounded-lg p-3 border mb-2`}>
      <Text className="text-xs text-muted mb-1">{label}</Text>
      <Text className="text-lg font-bold text-foreground">
        {value} {unit}
      </Text>
    </View>
  );
}

function CodeViewer({ title, code }: { title: string; code: string }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <View className="bg-surface rounded-lg border border-border overflow-hidden mb-3">
      <Pressable
        onPress={() => {
          setExpanded(!expanded);
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        }}
        className="p-4 flex-row justify-between items-center"
      >
        <Text className="text-sm font-semibold text-foreground">{title}</Text>
        <Text className="text-lg">{expanded ? "▼" : "▶"}</Text>
      </Pressable>
      {expanded && (
        <View className="bg-background p-3 border-t border-border">
          <Text className="text-xs font-mono text-muted leading-relaxed">
            {code}
          </Text>
        </View>
      )}
    </View>
  );
}

export default function VisionBackupScreen() {
  const colors = useColors();
  const [isCapturing, setIsCapturing] = useState(true);

  const visionCode = `# OpenCV + YOLOv8 Scoreboard Detection
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def capture_scoreboard(stream_url):
    cap = cv2.VideoCapture(stream_url)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Detect scoreboard region
        results = model(frame)
        
        # Extract ROI
        scoreboard = frame[y1:y2, x1:x2]
        
        # OCR extraction
        text = pytesseract.image_to_string(
            scoreboard,
            config='--psm 6'
        )
        
        process_ocr_data(text)`;

  const thermalCode = `# Thermal Management (GTX 1650 Ti)
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def monitor_thermal():
    while True:
        temp = pynvml.nvmlDeviceGetTemperature(
            handle, 0
        )
        util = pynvml.nvmlDeviceGetUtilizationRates(
            handle
        )
        
        if temp > 80:
            reduce_inference_fps()
        
        time.sleep(5)`;

  return (
    <ScreenContainer className="p-0">
      <ScrollView contentContainerStyle={{ flexGrow: 1 }}>
        {/* Header */}
        <View className="px-4 pt-4 pb-2">
          <Text className="text-2xl font-bold text-foreground">
            Vision Backup
          </Text>
          <Text className="text-sm text-muted mt-1">
            OpenCV + YOLOv8 scoreboard detection
          </Text>
        </View>

        {/* Capture Status */}
        <View className="px-4 py-4">
          <View className="bg-surface rounded-lg p-4 border border-border mb-4">
            <View className="flex-row justify-between items-center mb-3">
              <Text className="text-sm font-semibold text-foreground">
                Live Capture
              </Text>
              <Pressable
                onPress={() => {
                  setIsCapturing(!isCapturing);
                  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                }}
                className="bg-primary px-3 py-1 rounded-full"
              >
                <Text className="text-xs font-semibold text-background">
                  {isCapturing ? "Stop" : "Start"}
                </Text>
              </Pressable>
            </View>
            <View className="bg-background rounded-lg p-4 mb-3 h-40 items-center justify-center border border-border">
              <Text className="text-4xl mb-2">📹</Text>
              <Text className="text-xs text-muted text-center">
                {isCapturing
                  ? "Capturing scoreboard region..."
                  : "Capture stopped"}
              </Text>
            </View>
            {isCapturing && (
              <View className="flex-row items-center gap-2">
                <View className="w-2 h-2 rounded-full bg-success" />
                <Text className="text-xs text-muted">FPS: 30 | Latency: 45ms</Text>
              </View>
            )}
          </View>
        </View>

        {/* OCR Results */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            OCR Extracted Data
          </Text>
          {OCR_RESULTS.map((data, idx) => (
            <OCRCard key={idx} data={data} />
          ))}
        </View>

        {/* GPU Metrics */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            GPU Metrics (GTX 1650 Ti)
          </Text>
          <MetricCard
            label="Utilization"
            value={GPU_METRICS.utilization}
            unit="%"
            status="good"
          />
          <MetricCard
            label="Temperature"
            value={GPU_METRICS.temperature}
            unit="°C"
            status="good"
          />
          <MetricCard
            label="Memory"
            value={GPU_METRICS.memory}
            unit="MB"
            status="good"
          />
        </View>

        {/* Code Viewers */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Implementation
          </Text>
          <CodeViewer title="YOLOv8 Scoreboard Detection" code={visionCode} />
          <CodeViewer title="Thermal Management" code={thermalCode} />
        </View>
      </ScrollView>
    </ScreenContainer>
  );
}
