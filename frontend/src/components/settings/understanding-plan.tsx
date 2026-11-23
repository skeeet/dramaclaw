// SPDX-License-Identifier: Elastic-2.0
import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";

import { ChatGPTPlanPanel, useChatGPTPlan } from "@/components/settings/chatgpt-plan";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { queryKeys } from "@/lib/query-keys";
import type { UnderstandingPlanConfig } from "@/lib/queries/model-gateway";
import type { OkResponse } from "@/types/api";

const features = ["style_analysis", "image_reverse_prompt", "scene_headings"] as const;

export function UnderstandingPlanPanel({ saved, available }: {
  saved: UnderstandingPlanConfig;
  available: boolean;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState(saved.features);
  const [model, setModel] = useState(saved.model);
  const [effort, setEffort] = useState(saved.reasoningEffort);
  const plan = useChatGPTPlan(available && open && selected.length > 0);
  const requireImages = selected.some((feature) => feature !== "scene_headings");
  const modelChoice = plan.models.data?.find((item) => item.id === model
    && (!requireImages || item.supportsImages));
  const selectedEffort = modelChoice?.reasoningEfforts.includes(effort) ? effort : "";

  useEffect(() => {
    setSelected(saved.features);
    setModel(saved.model);
    setEffort(saved.reasoningEffort);
  }, [saved]);

  const save = useMutation({
    mutationFn: (config: UnderstandingPlanConfig) => api
      .post("api/v1/model-gateway/agent/chatgpt-plan/understanding", {
        json: config,
        headers: { "X-DramaClaw-Local": "1" },
      })
      .json<OkResponse<UnderstandingPlanConfig>>(),
    onSuccess() {
      void queryClient.invalidateQueries({ queryKey: queryKeys.modelGateway() });
      void queryClient.invalidateQueries({ queryKey: ["generation-credit-cost", "feature"] });
      toast.success(t("settings.modelConfig.understanding.saved"));
    },
    onError(error) {
      const detail = (error as { data?: { detail?: unknown } }).data?.detail;
      toast.error(typeof detail === "string" ? detail : t("settings.modelConfig.requestFailed"));
    },
  });

  return (
    <details
      className="mt-4 rounded-md border border-border/70 px-3 py-3"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer text-xs font-medium text-foreground">
        {t("settings.modelConfig.understanding.title")}
      </summary>
      <p className="mt-2 text-[11px] text-muted-foreground">
        {t("settings.modelConfig.understanding.description")}
      </p>
      <fieldset className="mt-3 flex flex-wrap gap-x-4 gap-y-2">
        <legend className="sr-only">{t("settings.modelConfig.understanding.title")}</legend>
        {features.map((feature) => (
          <label key={feature} className="flex items-center gap-2 text-xs">
            <input type="checkbox" checked={selected.includes(feature)} onChange={(event) => {
              setSelected(event.target.checked
                ? [...selected, feature]
                : selected.filter((item) => item !== feature));
            }} />
            {t(`settings.modelConfig.understanding.${feature}`)}
          </label>
        ))}
      </fieldset>
      {selected.length && available ? (
        <ChatGPTPlanPanel
          plan={plan} model={modelChoice?.id ?? ""} requireImages={requireImages}
          setModel={(value) => { setModel(value); setEffort(""); }}
          effort={selectedEffort} setEffort={setEffort}
        />
      ) : null}
      {!available && (
        <p className="mt-2 text-xs text-muted-foreground">
          {t("settings.modelConfig.understanding.unavailable")}
        </p>
      )}
      <div className="mt-3 flex justify-end">
        <Button
          type="button" size="sm"
          disabled={save.isPending || Boolean(selected.length && (
            !available || !plan.connected || !modelChoice || plan.connection.isPending
          ))}
          onClick={() => save.mutate({ features: selected, model, reasoningEffort: selectedEffort })}
        >
          {t("settings.modelConfig.understanding.save")}
        </Button>
      </div>
    </details>
  );
}
