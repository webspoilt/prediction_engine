import { ScrollView, Text, View, Pressable } from "react-native";
import { useState } from "react";
import { ScreenContainer } from "@/components/screen-container";
import { useColors } from "@/hooks/use-colors";
import * as Haptics from "expo-haptics";

interface DataSource {
  name: string;
  status: "connected" | "disconnected" | "degraded";
  lastUpdate: string;
  latency: number;
}

interface ScraperStatus {
  device: string;
  status: "active" | "standby" | "failed";
  uptime: string;
  dataPoints: number;
}

const DATA_SOURCES: DataSource[] = [
  {
    name: "Betfair WebSocket",
    status: "connected",
    lastUpdate: "2 seconds ago",
    latency: 45,
  },
  {
    name: "Cricbuzz API",
    status: "connected",
    lastUpdate: "3 seconds ago",
    latency: 120,
  },
  {
    name: "ESPN Cricket Feed",
    status: "degraded",
    lastUpdate: "8 seconds ago",
    latency: 250,
  },
];

const SCRAPER_STATUS: ScraperStatus[] = [
  {
    device: "Asus TUF (Primary)",
    status: "active",
    uptime: "23h 45m",
    dataPoints: 15240,
  },
  {
    device: "Poco X3 (Failover)",
    status: "standby",
    uptime: "12h 30m",
    dataPoints: 8120,
  },
];

function StatusBadge({ status }: { status: string }) {
  const colors = useColors();

  const statusConfig = {
    connected: { bg: "bg-success", text: "Connected" },
    disconnected: { bg: "bg-error", text: "Disconnected" },
    degraded: { bg: "bg-warning", text: "Degraded" },
    active: { bg: "bg-success", text: "Active" },
    standby: { bg: "bg-muted", text: "Standby" },
    failed: { bg: "bg-error", text: "Failed" },
  };

  const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.disconnected;

  return (
    <View className={`${config.bg} px-2 py-1 rounded-full`}>
      <Text className="text-xs font-semibold text-background">
        {config.text}
      </Text>
    </View>
  );
}

function DataSourceCard({ source }: { source: DataSource }) {
  return (
    <View className="bg-surface rounded-lg p-4 mb-3 border border-border">
      <View className="flex-row justify-between items-start mb-2">
        <Text className="text-sm font-semibold text-foreground flex-1">
          {source.name}
        </Text>
        <StatusBadge status={source.status} />
      </View>
      <View className="flex-row justify-between items-center gap-2">
        <View>
          <Text className="text-xs text-muted">Last Update</Text>
          <Text className="text-xs font-semibold text-foreground">
            {source.lastUpdate}
          </Text>
        </View>
        <View>
          <Text className="text-xs text-muted">Latency</Text>
          <Text className="text-xs font-semibold text-foreground">
            {source.latency}ms
          </Text>
        </View>
      </View>
    </View>
  );
}

function ScraperCard({ scraper }: { scraper: ScraperStatus }) {
  return (
    <View className="bg-surface rounded-lg p-4 mb-3 border border-border">
      <View className="flex-row justify-between items-start mb-3">
        <Text className="text-sm font-semibold text-foreground flex-1">
          {scraper.device}
        </Text>
        <StatusBadge status={scraper.status} />
      </View>
      <View className="flex-row gap-4">
        <View className="flex-1">
          <Text className="text-xs text-muted">Uptime</Text>
          <Text className="text-xs font-semibold text-foreground">
            {scraper.uptime}
          </Text>
        </View>
        <View className="flex-1">
          <Text className="text-xs text-muted">Data Points</Text>
          <Text className="text-xs font-semibold text-foreground">
            {scraper.dataPoints.toLocaleString()}
          </Text>
        </View>
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

export default function DataPipelineScreen() {
  const colors = useColors();

  const scraperCode = `# WebSocket Scraper (Betfair)
import asyncio
from playwright.async_api import async_playwright

async def sniff_websocket():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Listen to WebSocket messages
        async def handle_ws(ws):
            while True:
                msg = await ws.receive_text()
                process_ball_data(msg)
        
        page.on("websocket", handle_ws)
        await page.goto("https://betfair.com")
        await asyncio.sleep(3600)`;

  const fallbackCode = `# Failover Logic (Poco X3)
if primary_scraper_blocked():
    switch_to_fallover()
    start_poco_x3_server()
    sync_data_via_redis()
    
# Heartbeat check
def heartbeat_monitor():
    while True:
        if not ping_asus_tuf():
            activate_failover()
        time.sleep(30)`;

  return (
    <ScreenContainer className="p-0">
      <ScrollView contentContainerStyle={{ flexGrow: 1 }}>
        {/* Header */}
        <View className="px-4 pt-4 pb-2">
          <Text className="text-2xl font-bold text-foreground">
            Data Pipeline
          </Text>
          <Text className="text-sm text-muted mt-1">
            WebSocket scraping & redundancy
          </Text>
        </View>

        {/* Redundancy Status */}
        <View className="px-4 py-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Redundancy Status
          </Text>
          <View className="bg-primary/10 border border-primary rounded-lg p-4 mb-4">
            <View className="flex-row items-center gap-2 mb-2">
              <View className="w-3 h-3 rounded-full bg-success" />
              <Text className="text-sm font-semibold text-foreground">
                Primary + Fallover Active
              </Text>
            </View>
            <Text className="text-xs text-muted">
              Asus TUF is primary, Poco X3 on standby for failover
            </Text>
          </View>
        </View>

        {/* Scrapers */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Scrapers
          </Text>
          {SCRAPER_STATUS.map((scraper, idx) => (
            <ScraperCard key={idx} scraper={scraper} />
          ))}
        </View>

        {/* Data Sources */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Data Sources
          </Text>
          {DATA_SOURCES.map((source, idx) => (
            <DataSourceCard key={idx} source={source} />
          ))}
        </View>

        {/* Code Viewers */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Implementation
          </Text>
          <CodeViewer title="WebSocket Scraper" code={scraperCode} />
          <CodeViewer title="Failover Logic" code={fallbackCode} />
        </View>
      </ScrollView>
    </ScreenContainer>
  );
}
