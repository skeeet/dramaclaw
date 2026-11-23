// SPDX-License-Identifier: Elastic-2.0
// Copyright (c) 2026 ClaymoreLab
import { useId, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ExternalLink } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import type { OkResponse } from "@/types/api";

interface ChatGPTPlanStatus {
  status: "disconnected" | "connecting" | "connected" | "reauth_required";
  sharing: boolean;
  activeProfileId: string | null;
  profiles: Array<{ id: string; label: string; email: string }>;
  message: string | null;
}

interface PlanModel {
  id: string;
  label: string;
  reasoningEfforts: string[];
  defaultReasoningEffort: string;
  supportsImages?: boolean;
}

const path = "api/v1/model-gateway/agent/chatgpt-plan";
const headers = { "X-DramaClaw-Local": "1" };
const queryKey = ["director-chatgpt-plan"] as const;
const selectClass = "h-9 rounded-md border border-input bg-background px-2 text-sm";

function errorMessage(error: unknown, fallback: string) {
  const detail = (error as { data?: { detail?: unknown } })?.data?.detail;
  return typeof detail === "string" ? detail : fallback;
}

export function useChatGPTPlan(enabled: boolean) {
  const qc = useQueryClient();
  const status = useQuery({
    queryKey,
    enabled,
    retry: false,
    queryFn: async () => (await api.get(path, { headers }).json<OkResponse<ChatGPTPlanStatus>>()).data,
    refetchInterval: (query) => query.state.data?.status === "connecting" ? 1200 : false,
  });
  const connected = status.data?.status === "connected" && status.data.sharing;
  const models = useQuery({
    queryKey: [...queryKey, "models", status.data?.activeProfileId],
    enabled: enabled && connected,
    retry: false,
    queryFn: async () => {
      try {
        return (await api.get(path + "/models", { headers }).json<OkResponse<PlanModel[]>>()).data;
      } catch (error) {
        void qc.invalidateQueries({ queryKey, exact: true });
        throw new Error(errorMessage(error, "Could not load ChatGPT plan models."));
      }
    },
  });
  const connection = useMutation({
    mutationFn: async (input: { profileId: string | null; reconsent: boolean } | null) => {
      try {
        const response = await api.post(path + (input ? "/sign-in" : "/sign-out"), {
          headers,
          ...(input ? { json: input } : { timeout: 55_000 }),
        }).json<OkResponse<ChatGPTPlanStatus>>();
        return response.data;
      } catch (error) {
        throw new Error(errorMessage(error, "Could not update the ChatGPT plan connection."));
      }
    },
    onSuccess(data) {
      qc.setQueryData(queryKey, data);
      qc.removeQueries({ queryKey: [...queryKey, "models"] });
    },
  });
  return { status, models, connection, connected };
}

export function ChatGPTPlanPanel({
  plan, model, setModel, effort, setEffort, requireImages = false,
}: {
  plan: ReturnType<typeof useChatGPTPlan>;
  model: string;
  setModel: (model: string) => void;
  effort: string;
  setEffort: (effort: string) => void;
  requireImages?: boolean;
}) {
  const { t } = useTranslation();
  const accountId = useId();
  const modelId = useId();
  const effortId = useId();
  const [profileChoice, setProfileChoice] = useState<string>();
  const status = plan.status.data;
  const profile = status?.profiles.find(({ id }) => id === status.activeProfileId);
  const choice = profileChoice ?? status?.activeProfileId ?? "";
  const connecting = status?.status === "connecting";
  const needsConsent = status?.status === "connected" && !status.sharing;
  const reconsent = needsConsent && choice === status?.activeProfileId;
  const message = plan.connection.error?.message || status?.message || plan.models.error?.message;
  const models = plan.models.data?.filter((entry) => !requireImages || entry.supportsImages);
  const selected = models?.find((entry) => entry.id === model);
  const statusText = connecting
    ? t("settings.modelConfig.agent.chatgpt.connecting")
    : plan.connected
      ? t("settings.modelConfig.agent.chatgpt.connected", { account: profile?.email || profile?.label })
      : status?.status === "reauth_required"
        ? t("settings.modelConfig.agent.chatgpt.reauthorize")
        : needsConsent
          ? t("settings.modelConfig.agent.chatgpt.needsConsent")
          : t("settings.modelConfig.agent.chatgpt.disconnected");

  return (
    <div className="mt-3 space-y-2.5">
      <p className="text-xs text-muted-foreground" role="status">
        {plan.status.isPending
          ? t("settings.modelConfig.agent.chatgpt.checking")
          : plan.status.isError
            ? t("settings.modelConfig.agent.chatgpt.unavailable")
            : statusText}
      </p>
      {status?.profiles.length ? (
        <div className="flex flex-wrap items-center gap-2">
          <Label htmlFor={accountId}>{t("settings.modelConfig.agent.chatgpt.account")}</Label>
          <select
            id={accountId}
            className={selectClass}
            value={choice}
            disabled={connecting || plan.connection.isPending}
            onChange={(event) => setProfileChoice(event.target.value)}
          >
            {status.profiles.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.label}{entry.email && " · " + entry.email}
              </option>
            ))}
            <option value="">{t("settings.modelConfig.agent.chatgpt.addAccount")}</option>
          </select>
        </div>
      ) : null}
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          size="sm"
          disabled={!status || connecting || plan.connection.isPending}
          onClick={() => {
            plan.connection.mutate({ profileId: choice || null, reconsent });
            setProfileChoice(undefined);
          }}
        >
          {t(reconsent
            ? "settings.modelConfig.agent.chatgpt.enable"
            : "settings.modelConfig.agent.chatgpt.signIn")}
        </Button>
        {status?.status === "connected" ? (
          <Button type="button" size="sm" variant="outline" disabled={plan.connection.isPending}
            onClick={() => plan.connection.mutate(null)}>
            {t("settings.modelConfig.agent.chatgpt.signOut")}
          </Button>
        ) : null}
        <a href="https://chatgpt.com/settings/usage" target="_blank" rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-xs text-primary hover:underline">
          {t("settings.modelConfig.agent.chatgpt.usage")} <ExternalLink className="size-3" />
        </a>
      </div>
      {message ? <p className="text-xs text-destructive" role="alert">{message}</p> : null}
      {plan.connected ? (
        <div className="flex flex-wrap items-center gap-2">
          <Label htmlFor={modelId}>{t("settings.modelConfig.agent.model")}</Label>
          <select id={modelId} className={selectClass} value={model}
            onChange={(event) => setModel(event.target.value)}
            disabled={!models?.length || plan.connection.isPending}>
            <option value="" disabled>{t("settings.modelConfig.agent.chatgpt.chooseModel")}</option>
            {models?.map((option) => (
              <option value={option.id} key={option.id}>{option.label}</option>
            ))}
          </select>
          {selected?.reasoningEfforts.length ? (
            <>
              <Label htmlFor={effortId}>{t("settings.modelConfig.agent.chatgpt.reasoningEffort")}</Label>
              <select id={effortId} className={selectClass} value={effort}
                onChange={(event) => setEffort(event.target.value)}
                disabled={plan.connection.isPending}>
                <option value="">{t("settings.modelConfig.agent.chatgpt.modelDefault")}</option>
                {selected.reasoningEfforts.map((value) => (
                  <option value={value} key={value}>
                    {t(`settings.modelConfig.agent.chatgpt.effort.${value}`, { defaultValue: value })}
                  </option>
                ))}
              </select>
            </>
          ) : null}
          {plan.models.isError ? (
            <Button type="button" size="sm" variant="outline" onClick={() => void plan.models.refetch()}>
              {t("settings.modelConfig.agent.chatgpt.retryModels")}
            </Button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
