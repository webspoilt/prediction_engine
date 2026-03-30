import { ScrollView, Text, View, Pressable, FlatList } from "react-native";
import { useState, useEffect } from "react";
import { ScreenContainer } from "@/components/screen-container";
import { useColors } from "@/hooks/use-colors";
import * as Haptics from "expo-haptics";

interface Match {
  id: string;
  teamA: string;
  teamB: string;
  scoreA: number;
  scoreB: number;
  wicketsA: number;
  wicketsB: number;
  oversA: string;
  oversB: string;
  status: "live" | "upcoming" | "completed";
  winProbability: number;
  crr: number;
  rrr: number;
  lastUpdate: string;
}

const MOCK_LIVE_MATCH: Match = {
  id: "match_001",
  teamA: "Mumbai Indians",
  teamB: "Chennai Super Kings",
  scoreA: 156,
  scoreB: 0,
  wicketsA: 3,
  wicketsB: 0,
  oversA: "18.3",
  oversB: "0.0",
  status: "live",
  winProbability: 72,
  crr: 8.5,
  rrr: 9.2,
  lastUpdate: "2 seconds ago",
};

const MOCK_UPCOMING_MATCHES: Match[] = [
  {
    id: "match_002",
    teamA: "Rajasthan Royals",
    teamB: "Delhi Capitals",
    scoreA: 0,
    scoreB: 0,
    wicketsA: 0,
    wicketsB: 0,
    oversA: "0.0",
    oversB: "0.0",
    status: "upcoming",
    winProbability: 0,
    crr: 0,
    rrr: 0,
    lastUpdate: "Starts in 2 hours",
  },
  {
    id: "match_003",
    teamA: "Kolkata Knight Riders",
    teamB: "Punjab Kings",
    scoreA: 0,
    scoreB: 0,
    wicketsA: 0,
    wicketsB: 0,
    oversA: "0.0",
    oversB: "0.0",
    status: "upcoming",
    winProbability: 0,
    crr: 0,
    rrr: 0,
    lastUpdate: "Starts in 4 hours",
  },
];

function WinProbabilityGauge({ probability }: { probability: number }) {
  const colors = useColors();
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (probability / 100) * circumference;

  return (
    <View className="items-center justify-center gap-2">
      <View className="relative w-32 h-32 items-center justify-center">
        <svg width="128" height="128" viewBox="0 0 128 128">
          {/* Background circle */}
          <circle
            cx="64"
            cy="64"
            r="45"
            fill="none"
            stroke={colors.border}
            strokeWidth="6"
          />
          {/* Progress circle */}
          <circle
            cx="64"
            cy="64"
            r="45"
            fill="none"
            stroke={probability > 50 ? colors.success : colors.warning}
            strokeWidth="6"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            transform="rotate(-90 64 64)"
          />
        </svg>
        <View className="absolute items-center justify-center">
          <Text className="text-3xl font-bold text-foreground">
            {probability}%
          </Text>
          <Text className="text-xs text-muted">Win Probability</Text>
        </View>
      </View>
    </View>
  );
}

function LiveMatchCard({ match }: { match: Match }) {
  const colors = useColors();

  return (
    <Pressable
      onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
      style={({ pressed }) => [
        {
          opacity: pressed ? 0.7 : 1,
        },
      ]}
      className="bg-surface rounded-2xl p-4 mb-4 border border-border"
    >
      {/* Status Badge */}
      <View className="flex-row justify-between items-center mb-3">
        <View className="bg-primary px-3 py-1 rounded-full">
          <Text className="text-xs font-semibold text-background">
            {match.status.toUpperCase()}
          </Text>
        </View>
        <Text className="text-xs text-muted">{match.lastUpdate}</Text>
      </View>

      {/* Teams and Scores */}
      <View className="gap-3 mb-4">
        {/* Team A */}
        <View className="flex-row justify-between items-center">
          <Text className="text-sm font-semibold text-foreground flex-1">
            {match.teamA}
          </Text>
          <View className="flex-row items-center gap-2">
            <Text className="text-lg font-bold text-foreground">
              {match.scoreA}
            </Text>
            <Text className="text-xs text-muted">
              {match.wicketsA}w {match.oversA}
            </Text>
          </View>
        </View>

        {/* Divider */}
        <View className="h-px bg-border" />

        {/* Team B */}
        <View className="flex-row justify-between items-center">
          <Text className="text-sm font-semibold text-foreground flex-1">
            {match.teamB}
          </Text>
          <View className="flex-row items-center gap-2">
            <Text className="text-lg font-bold text-foreground">
              {match.scoreB}
            </Text>
            <Text className="text-xs text-muted">
              {match.wicketsB}w {match.oversB}
            </Text>
          </View>
        </View>
      </View>

      {/* Win Probability Gauge */}
      <View className="mb-4">
        <WinProbabilityGauge probability={match.winProbability} />
      </View>

      {/* Quick Stats */}
      <View className="flex-row gap-2">
        <View className="flex-1 bg-background rounded-lg p-2">
          <Text className="text-xs text-muted">CRR</Text>
          <Text className="text-sm font-semibold text-foreground">
            {match.crr}
          </Text>
        </View>
        <View className="flex-1 bg-background rounded-lg p-2">
          <Text className="text-xs text-muted">RRR</Text>
          <Text className="text-sm font-semibold text-foreground">
            {match.rrr}
          </Text>
        </View>
      </View>
    </Pressable>
  );
}

function UpcomingMatchCard({ match }: { match: Match }) {
  const colors = useColors();

  return (
    <Pressable
      onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
      style={({ pressed }) => [
        {
          opacity: pressed ? 0.7 : 1,
        },
      ]}
      className="bg-surface rounded-xl p-3 mr-3 border border-border min-w-48"
    >
      <View className="flex-row justify-between items-start mb-2">
        <View className="flex-1">
          <Text className="text-xs font-semibold text-primary mb-1">
            {match.teamA}
          </Text>
          <Text className="text-xs font-semibold text-primary">
            vs {match.teamB}
          </Text>
        </View>
      </View>
      <Text className="text-xs text-muted mt-2">{match.lastUpdate}</Text>
    </Pressable>
  );
}

export default function DashboardScreen() {
  const colors = useColors();
  const [liveMatch, setLiveMatch] = useState(MOCK_LIVE_MATCH);
  const [upcomingMatches, setUpcomingMatches] = useState(MOCK_UPCOMING_MATCHES);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    // Simulate data refresh
    setTimeout(() => {
      setLiveMatch((prev) => ({
        ...prev,
        winProbability: Math.max(
          Math.min(prev.winProbability + Math.random() * 10 - 5, 99),
          1
        ),
        lastUpdate: "just now",
      }));
      setIsRefreshing(false);
    }, 500);
  };

  return (
    <ScreenContainer className="p-0">
      <ScrollView contentContainerStyle={{ flexGrow: 1 }}>
        {/* Header */}
        <View className="px-4 pt-4 pb-2 flex-row justify-between items-start">
          <View className="flex-1">
            <Text className="text-3xl font-bold text-foreground">
              IPL Win Predictor
            </Text>
            <Text className="text-sm text-muted mt-1">
              Real-time match predictions
            </Text>
          </View>
          <Pressable
            onPress={handleRefresh}
            disabled={isRefreshing}
            style={({ pressed }) => [{ opacity: pressed ? 0.6 : 1 }]}
            className="ml-2"
          >
            <Text className="text-2xl">🔄</Text>
          </Pressable>
        </View>

        {/* Live Match Section */}
        <View className="px-4 py-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Live Match
          </Text>
          <LiveMatchCard match={liveMatch} />
        </View>

        {/* System Health Section */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            System Health
          </Text>
          <View className="flex-row gap-2">
            <View className="flex-1 bg-surface rounded-lg p-3 border border-border">
              <View className="flex-row items-center gap-2 mb-1">
                <View className="w-2 h-2 rounded-full bg-success" />
                <Text className="text-xs text-muted">Data Pipeline</Text>
              </View>
              <Text className="text-xs font-semibold text-foreground">
                Connected
              </Text>
            </View>
            <View className="flex-1 bg-surface rounded-lg p-3 border border-border">
              <View className="flex-row items-center gap-2 mb-1">
                <View className="w-2 h-2 rounded-full bg-success" />
                <Text className="text-xs text-muted">Vision Backup</Text>
              </View>
              <Text className="text-xs font-semibold text-foreground">
                Active
              </Text>
            </View>
            <View className="flex-1 bg-surface rounded-lg p-3 border border-border">
              <View className="flex-row items-center gap-2 mb-1">
                <View className="w-2 h-2 rounded-full bg-success" />
                <Text className="text-xs text-muted">ML Model</Text>
              </View>
              <Text className="text-xs font-semibold text-foreground">
                Ready
              </Text>
            </View>
          </View>
        </View>

        {/* Upcoming Matches Section */}
        <View className="pb-4">
          <View className="px-4 mb-3">
            <Text className="text-lg font-semibold text-foreground">
              Upcoming Matches
            </Text>
          </View>
          <FlatList
            data={upcomingMatches}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => <UpcomingMatchCard match={item} />}
            horizontal
            scrollEnabled
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={{ paddingHorizontal: 16 }}
          />
        </View>
      </ScrollView>
    </ScreenContainer>
  );
}
