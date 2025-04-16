/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable react-hooks/exhaustive-deps */
"use client";
import React, { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { LoaderCore } from "./ui/multi-step-loader";
import CustomButton from "./ui/CustomButton";
import { Globe, Upload, Video } from "lucide-react";
import { cn } from "../lib/utils";
import { PlaceholdersAndVanishInput } from "./ui/placeholders-and-vanish-input";
import axios from "axios";
import { IKUpload } from "imagekitio-next";
import toast from "react-hot-toast";
import { getModelToken } from "../actions/modelToken";
import { useSession } from "next-auth/react";
import { SavePrismaChats } from "../actions/saveChat";

export interface ModelResponse {
  status: string;
  results: {
    is_fake: boolean;
    confidence: number;
    model_index: number;
  }[];
}

const authenticator = async () => {
  try {
    const res = await fetch(`/api/upload/video`);
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(
        `Request failed with code ${res.status} and ${errorText}`,
      );
    }
    const data = await res.json();
    const { signature, expire, token } = data;
    return { signature, expire, token };
  } catch (error) {
    throw new Error(`Failed ${error}`);
  }
};

const loadingStates = [
  {
    text: "Uploading your video ",
  },
  {
    text: "Sending to AI-Model",
  },
  {
    text: "Preparing for analysis",
  },
  {
    text: "Analyzing video with AI",
  },
  {
    text: "Extracting key insights",
  },
  {
    text: "Processing results",
  },
  {
    text: "Cleaning up for data privacy",
  },
  {
    text: "Reviewing and saving output",
  },
  {
    text: "Completed successfully",
  },
];

const placeholders = [
  "Paste a blog, news, url",
  "Drop a link — we'll fact-check it for you",
  "Is this news real or fake?",
  "Enter a suspicious link to uncover the truth",
  "Check if that viral post is legit",
];

function isValidURL(url: string): boolean {
  const urlRegex = /^(https?:\/\/)?([\w-]+\.)+[\w-]{2,}(\/[\w-./?%&=]*)?$/i;
  return urlRegex.test(url);
}

export default function Message() {
  const [url, setUrl] = useState("");
  const [notValidUrl, setNotValidUrl] = useState<boolean>(false);
  const [multiSteploading, setMultiStepLoading] = useState(false);
  const [currentState, setCurrentState] = useState(0);
  const session = useSession()
  const [selectedOption, setSelectedOption] = useState<"news" | "deepfake">(
    "deepfake",
  );

  const uploadRef = useRef<HTMLInputElement | null>(null);
  const [modelResponse, setModelResponse] = useState<ModelResponse | null>(null);

  const [videoUrl, setVideoUrl] = useState("");
  // storing for after response deleting the video
  const [fileId, setFileId] = useState("");

  const handleOption = (opt: "news" | "deepfake") => {
    setSelectedOption(opt);
  };

  const [deepFakeSelectedOption, setDeepfakeSelectedOption] = useState<
    "quick" | "deep"
  >("quick");


  const handleAfterUpload = async () => {
    try {
      if (!fileId || !videoUrl) return;
      setCurrentState(1);
      await new Promise((resolve) => setTimeout(resolve, 300));
      setCurrentState(2);
      let timer: NodeJS.Timeout;
      if (deepFakeSelectedOption === "quick") {
        timer = setInterval(() => {
          if (currentState < 5) {
            setCurrentState((c) => c + 1);
          }
        }, 5000);
        setTimeout(() => {
          clearInterval(timer);
        }, 16000);
      }
      if (deepFakeSelectedOption === "deep") {
        timer = setInterval(() => {
          if (currentState < 5) {
            setCurrentState((c) => c + 1);
          }
        }, 6000);
        setTimeout(() => {
          clearInterval(timer);
        }, 19000);
      }
      const tempToken = await getModelToken();
      const res = await axios.post(
        `https://trufakemodel.ashishtiwari.net/predict`,
        {
          method: deepFakeSelectedOption,
          video_url:videoUrl,
          token: tempToken,
        },
      );
      setCurrentState(6);
      const deletingVideo = await axios.post("/api/delete/video", {
        fileId: fileId,
      });

      setCurrentState(7);
      setCurrentState(6);
      setModelResponse(res.data);
      // storing in db
      const videoTitle = videoUrl.split("jula420ykc/")[1]?.split('/')[0];
      console.log("VIDEO TITLE IS",videoTitle)
      const dbRes = await SavePrismaChats({
        title: videoTitle || "",
        type: selectedOption,
        modelResponse: JSON.stringify(res.data),
      })
      console.log("ADDED TO DB", dbRes)
      setCurrentState(8);
      setMultiStepLoading(false);
    } catch (error) {
      await axios.post("/api/delete/video", {
        fileId: fileId,
      });
      console.log("ERROR NEW ", error)
      toast.error("Something went wrong")
      setMultiStepLoading(false)
    }
  };
  useEffect(() => {
    if (fileId && videoUrl) {
      handleAfterUpload();
    }
  }, [fileId, videoUrl]);

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUrl(e.target.value);
    if (e.target.value.length > 0 && !isValidURL(e.target.value)) {
      setNotValidUrl(true);
    } else {
      setNotValidUrl(false);
    }
  };
  const onUrlSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    console.log("submitted");
  };

  return (
    <div className="h-full w-full">
      {/* multi loader here  */}
      {multiSteploading && (
        <AnimatePresence mode="wait">
          <motion.div
            initial={{
              opacity: 0,
            }}
            animate={{
              opacity: 1,
            }}
            exit={{
              opacity: 0,
            }}
            className="fixed inset-0 z-[100] flex h-full w-full items-center justify-center bg-black backdrop-blur-2xl"
          >
            <div className="relative h-96">
              <LoaderCore value={currentState} loadingStates={loadingStates} />
            </div>

            <div className="absolute inset-x-0 bottom-0 z-20 h-full bg-white bg-gradient-to-t [mask-image:radial-gradient(900px_at_center,transparent_30%,white)] dark:bg-black" />
          </motion.div>
        </AnimatePresence>
      )}

      {/* main content */}

      <div className="flex flex-col items-center pt-20">
        
        {modelResponse === null && (
          <div className="space-x-10">
            <CustomButton
              isLoading={false}
              Icon={Video}
              className={cn(
                "rounded-3xl border border-neutral-700 bg-transparent px-10 py-6",
                selectedOption === "deepfake" &&
                  "border-0 bg-blue-400/35 text-blue-400",
              )}
              iconStyle="mr-1"
              onClick={() => {
                handleOption("deepfake");
              }}
            >
              Deepfake
            </CustomButton>
            <CustomButton
              isLoading={false}
              Icon={Globe}
              className={cn(
                "rounded-3xl border border-neutral-700 bg-transparent px-10 py-6",
                selectedOption === "news" &&
                  "border-0 bg-blue-400/35 text-blue-400",
              )}
              iconStyle="mr-1"
              onClick={() => {
                handleOption("news");
              }}
            >
              News
            </CustomButton>
          </div>
        )}

        <div className="transition-all duration-400">
          {selectedOption === "news" && modelResponse === null ? (
            <div className="flex h-[40rem] flex-col items-center px-4 pt-10 transition-all duration-400">
              <h2 className="mb-10 animate-pulse text-center text-5xl text-neutral-600">
                COMING SOON
              </h2>
              {/* it is disabled inside */}
              <PlaceholdersAndVanishInput
                placeholders={placeholders}
                onChange={handleUrlChange}
                onSubmit={onUrlSubmit}
              />
              {notValidUrl && (
                <h4 className="text-danger pt-1 pl-4 select-none">
                  paste valid url link
                </h4>
              )}
            </div>
          ) : modelResponse === null ? (
            <div className="mt-12 flex flex-col items-center transition-discrete duration-400">
              <IKUpload
                className="hidden"
                ref={uploadRef}
                validateFile={(file) => {
                  if (file.size > 1024 * 1024 * 100) {
                    // file should less than 100MB
                    toast.error("File should less than 100mb", {
                      duration: 2000,
                    });
                    return false;
                  }
                  const allowedTypes = ["video/mp4", "video/quicktime"];

                  if (!allowedTypes.includes(file.type)) {
                    toast.error("Only mp4 are allowed", {
                      duration: 2000,
                    });
                    return false;
                  }
                  return true;
                }}
                onSuccess={(res) => {
                  console.log("HI", res)
                  setFileId(res.fileId);
                  setVideoUrl(res.url);
                  handleAfterUpload();
                }}
                onUploadProgress={(progress) => {
                  console.log("PROGRESS", progress);
                }}
                onUploadStart={(det) => {
                  setMultiStepLoading(true);
                }}
                urlEndpoint={`${process.env.NEXT_PUBLIC_URL_ENDPOINT}`}
                publicKey={`${process.env.NEXT_PUBLIC_KEY}`}
                authenticator={authenticator}
              />
              <div className="flex flex-col">
                {/* box */}
                <div
                  onClick={() => {
                    if (uploadRef.current) {
                      uploadRef.current.click();
                    }
                  }}
                  className="group flex h-32 w-64 cursor-pointer flex-col items-center justify-center rounded-3xl bg-neutral-700/40 transition-all duration-300 hover:bg-neutral-800"
                >
                  <h2 className="relative bottom-4 font-semibold text-white/70">
                    Upload video
                  </h2>
                  <h4 className="relative bottom-3 text-white/50">
                    click to upload
                  </h4>
                  <Upload className="size-6 transition-all duration-300 group-hover:scale-110 group-hover:opacity-50" />
                </div>
              </div>
              {/* deep and quick buttons */}
              <div className="space-x-4 pt-8">
                <CustomButton
                  isLoading={false}
                  Icon={null}
                  className={cn(
                    "px rounded-3xl border border-neutral-700 bg-transparent",
                    deepFakeSelectedOption === "quick" &&
                      "border-1 bg-neutral-700/50 font-semibold text-white active:scale-90",
                  )}
                  iconStyle="mr-1"
                  onClick={() => {
                    setDeepfakeSelectedOption("quick");
                  }}
                >
                  Quick
                </CustomButton>
                <CustomButton
                  isLoading={false}
                  Icon={null}
                  className={cn(
                    "rounded-3xl border border-neutral-700 bg-transparent",
                    deepFakeSelectedOption === "deep" &&
                      "border-1 bg-neutral-700/50 font-semibold text-white active:scale-90",
                  )}
                  iconStyle="mr-1"
                  onClick={() => {
                    setDeepfakeSelectedOption("deep");
                  }}
                >
                  Deep
                </CustomButton>
              </div>
            </div>
          ) : null}
        </div>
        {modelResponse && (
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

            <div className="flex justify-center pt-4">
              <a
              href="/dashboard"
                className="rounded-2xl border py-2 px-4 transition-opacity duration-300 border-neutral-800 hover:opacity-60 bg-neutral-800 text-red-400 "
              >
                Clear Results
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
