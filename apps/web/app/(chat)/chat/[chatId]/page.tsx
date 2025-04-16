import React from "react";
import { GetChatWithId } from "../../../../actions/getChatWithId";
import { Chats } from "../../../../components/Sidebar";
import { cn } from "../../../../lib/utils";
import { ModelResponse } from "../../../../components/Message";

interface PageProps {
  params: Promise<{
    chatId: string;
  }>;
}

export default async function page({ params }: PageProps) {
  const { chatId } = await params;
  const data = await GetChatWithId(chatId);
  const parseData = JSON.parse(data.toString()) as Chats;
  const modelResponse = JSON.parse(parseData.modelResponse) as ModelResponse;

  return (
    <div className="flex w-full justify-center items-center pt-10 ">
      <div className="w-full max-w-2xl items-center justify-center space-y-6 rounded-3xl border border-white/10 bg-black/10 px-6 py-8 shadow-lg">
        <h2 className="text-center text-3xl font-semibold text-white">
          Deepfake Results
        </h2>

        <div className="space-y-4">
          {modelResponse.results.map((res) => {
            const confidenceColor =
              res.confidence > 0.8
                ? res.is_fake
                  ? "text-red-400"
                  : "text-green-400"
                : res.confidence > 0.5
                  ? "text-yellow-400"
                  : "text-neutral-400";

            return (
              <div
                key={res.model_index}
                className="items-center justify-center rounded-2xl border border-white/10 bg-black/10 p-5 backdrop-blur-sm transition-all hover:scale-[1.015]"
              >
                <div className="flex items-center justify-between">
                  <span className="text-lg font-medium text-white">
                    Model #{res.model_index + 1}
                  </span>
                  <span
                    className={cn(
                      "rounded-full px-3 py-1 text-sm font-semibold",
                      res.is_fake
                        ? "bg-red-500/10 text-red-400"
                        : "bg-green-500/10 text-green-400",
                    )}
                  >
                    {res.confidence > 0.5 ? "Fake" : "Real"}
                  </span>
                </div>

                <div className="mt-2 text-sm text-white/60">
                  Confidence:{" "}
                  <span className={`${confidenceColor} font-bold`}>
                    {res.confidence > 0.5
                      ? (res.confidence * 100).toFixed(1)
                      : ((1 - res.confidence) * 100).toFixed(1)}
                    %
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
