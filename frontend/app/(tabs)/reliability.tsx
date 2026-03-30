import React, { useState } from "react";
import { ScrollView, Text, View, Pressable } from "react-native";
import { ScreenContainer } from "@/components/screen-container";
import { useColors } from "@/hooks/use-colors";
import * as Haptics from "expo-haptics";

interface ProcessStatus {
  name: string;
  status: "running" | "stopped" | "error";
  uptime: string;
  memory: number;
  cpu: number;
}

interface SystemMetric {
  label: string;
  value: number;
  unit: string;
}

const PROCESS_STATUS: ProcessStatus[] = [
  {
    name: "Data Scraper (Asus TUF)",
    status: "running",
    uptime: "23h 45m",
    memory: 245,
    cpu: 18,
  },
  {
    name: "Vision Pipeline (GPU)",
    status: "running",
    uptime: "18h 30m",
    memory: 1820,
    cpu: 62,
  },
  {
    name: "ML Model Server",
    status: "running",
    uptime: "45h 20m",
    memory: 512,
    cpu: 8,
  },
  {
    name: "Redis Cache",
    status: "running",
    uptime: "72h 15m",
    memory: 380,
    cpu: 2,
  },
];

const SYSTEM_METRICS: SystemMetric[] = [
  { label: "Uptime", value: 72, unit: "h" },
  { label: "Heartbeat Latency", value: 45, unit: "ms" },
  { label: "Memory Usage", value: 62, unit: "%" },
  { label: "CPU Usage", value: 28, unit: "%" },
];

function StatusBadge({ status }: { status: string }) {
  const statusConfig = {
    running: { bg: "bg-success", text: "Running" },
    stopped: { bg: "bg-muted", text: "Stopped" },
    error: { bg: "bg-error", text: "Error" },
  };

  const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.stopped;

  return (
    <View className={`${config.bg} px-2 py-1 rounded-full`}>
      <Text className="text-xs font-semibold text-background">
        {config.text}
      </Text>
    </View>
  );
}

function ProcessCard({ process }: { process: ProcessStatus }) {
  return (
    <View className="bg-surface rounded-lg p-4 mb-3 border border-border">
      <View className="flex-row justify-between items-start mb-3">
        <View className="flex-1">
          <Text className="text-sm font-semibold text-foreground">
            {process.name}
          </Text>
        </View>
        <StatusBadge status={process.status} />
      </View>

      <View className="flex-row gap-3">
        <View className="flex-1">
          <Text className="text-xs text-muted mb-1">Uptime</Text>
          <Text className="text-xs font-semibold text-foreground">
            {process.uptime}
          </Text>
        </View>
        <View className="flex-1">
          <Text className="text-xs text-muted mb-1">Memory</Text>
          <Text className="text-xs font-semibold text-foreground">
            {process.memory} MB
          </Text>
        </View>
        <View className="flex-1">
          <Text className="text-xs text-muted mb-1">CPU</Text>
          <Text className="text-xs font-semibold text-foreground">
            {process.cpu}%
          </Text>
        </View>
      </View>
    </View>
  );
}

function HeartbeatIndicator() {
  const [pulse, setPulse] = useState(true);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setPulse((prev) => !prev);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <View className="flex-row items-center gap-3">
      <View className="relative w-12 h-12">
        <View className="absolute inset-0 items-center justify-center">
          <View
            className={`w-3 h-3 rounded-full ${
              pulse ? "bg-success" : "bg-success/50"
            }`}
          />
        </View>
        {pulse && (
          <View className="absolute inset-0 items-center justify-center">
            <View className="w-6 h-6 rounded-full border-2 border-success opacity-50" />
          </View>
        )}
      </View>
      <View className="flex-1">
        <Text className="text-sm font-semibold text-foreground">
          Heartbeat Active
        </Text>
        <Text className="text-xs text-muted">Asus TUF ↔ Poco X3</Text>
      </View>
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

export default function ReliabilityScreen() {
  const colors = useColors();
  const [selectedProcess, setSelectedProcess] = useState<string | null>(null);

  const redisCode = `# Redis State Management
import redis

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)

# Store match state
redis_client.set(
    f'match:{match_id}:state',
    json.dumps(match_data),
    ex=3600  # 1 hour expiry
)

# Pub/Sub for real-time updates
pubsub = redis_client.pubsub()
pubsub.subscribe('match_updates')`;

  const pm2Code = `# PM2 Process Management
module.exports = {
  apps: [
    {
      name: 'data-scraper',
      script: 'scraper.py',
      instances: 1,
      exec_mode: 'cluster',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      error_file: './logs/error.log',
      out_file: './logs/out.log',
    },
    {
      name: 'ml-server',
      script: 'ml_server.py',
      instances: 2,
      exec_mode: 'cluster',
      autorestart: true,
    },
  ],
};`;

  const heartbeatCode = `# Heartbeat Mechanism
import socket
import time

def heartbeat_sender(target_ip, port):
    while True:
        try:
            sock = socket.socket()
            sock.connect((target_ip, port))
            sock.send(b'PING')
            response = sock.recv(1024)
            
            if response == b'PONG':
                redis_client.set(
                    'heartbeat:asus_tuf',
                    'alive',
                    ex=60
                )
        except Exception as e:
            activate_failover()
        finally:
            sock.close()
            time.sleep(30)`;

  return (
    <ScreenContainer className="p-0">
      <ScrollView contentContainerStyle={{ flexGrow: 1 }}>
        {/* Header */}
        <View className="px-4 pt-4 pb-2">
          <Text className="text-2xl font-bold text-foreground">
            System Health
          </Text>
          <Text className="text-sm text-muted mt-1">
            Redis, PM2, and heartbeat monitoring
          </Text>
        </View>

        {/* Heartbeat Status */}
        <View className="px-4 py-4">
          <View className="bg-success/10 border border-success rounded-lg p-4 mb-4">
            <HeartbeatIndicator />
          </View>
        </View>

        {/* System Metrics */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            System Metrics
          </Text>
          <View className="flex-row gap-2 flex-wrap">
            {SYSTEM_METRICS.map((metric, idx) => (
              <View key={idx} className="flex-1 min-w-24 bg-surface rounded-lg p-3 border border-border">
                <Text className="text-xs text-muted mb-1">{metric.label}</Text>
                <Text className="text-lg font-bold text-primary">
                  {metric.value}
                  <Text className="text-xs text-muted">{metric.unit}</Text>
                </Text>
              </View>
            ))}
          </View>
        </View>

        {/* Processes */}
        <View className="px-4 pb-4">
          <View className="flex-row justify-between items-center mb-3">
            <Text className="text-lg font-semibold text-foreground">
              PM2 Processes
            </Text>
            <Pressable
              onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              }}
              className="bg-primary px-3 py-1 rounded-full"
            >
              <Text className="text-xs font-semibold text-background">
                Restart All
              </Text>
            </Pressable>
          </View>
          {PROCESS_STATUS.map((process, idx) => (
            <ProcessCard key={idx} process={process} />
          ))}
        </View>

        {/* Redis Status */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Redis Cache
          </Text>
          <View className="bg-surface rounded-lg p-4 border border-border mb-4">
            <View className="flex-row justify-between items-center mb-3">
              <View>
                <Text className="text-sm font-semibold text-foreground">
                  Connection Status
                </Text>
                <Text className="text-xs text-muted mt-1">
                  localhost:6379
                </Text>
              </View>
              <View className="w-3 h-3 rounded-full bg-success" />
            </View>
            <View className="flex-row gap-2">
              <View className="flex-1">
                <Text className="text-xs text-muted mb-1">Keys Stored</Text>
                <Text className="text-lg font-bold text-foreground">
                  1,245
                </Text>
              </View>
              <View className="flex-1">
                <Text className="text-xs text-muted mb-1">Memory</Text>
                <Text className="text-lg font-bold text-foreground">
                  3.2 MB
                </Text>
              </View>
            </View>
          </View>
        </View>

        {/* Code Viewers */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Implementation
          </Text>
          <CodeViewer title="Redis State Management" code={redisCode} />
          <CodeViewer title="PM2 Configuration" code={pm2Code} />
          <CodeViewer title="Heartbeat Mechanism" code={heartbeatCode} />
        </View>
      </ScrollView>
    </ScreenContainer>
  );
}


