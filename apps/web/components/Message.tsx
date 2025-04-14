/* eslint-disable react-hooks/exhaustive-deps */
"use client";
import React, { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { LoaderCore } from "./ui/multi-step-loader";
import CustomButton from "./ui/CustomButton";
import { Globe, Upload, Video } from "lucide-react";
import { cn } from "../lib/utils";
import { PlaceholdersAndVanishInput } from "./ui/placeholders-and-vanish-input";
import axios from "axios"

interface ModelResponse {
  videoId: string;
  isFake: boolean;
  confidence: number;
}


import { IKUpload } from "imagekitio-next";
import toast from "react-hot-toast";

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
    text: "Uploading a video",
  },
  {
    text: "Sending to model",
  },
  {
    text: "Model working on video",
  },
  {
    text: "Got response from model",
  },
  {
    text: "Maintaining privacy (deleting uploaded video)",
  },
  {
    text: "Evaluting response",
  },
  {
    text: "Finished",
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
  const [multiSteploading, setMultiStepLoading] = useState(false);
  const [currentState, setCurrentState] = useState(0);
  const [selectedOption, setSelectedOption] = useState<"news" | "deepfake">(
    "news",
  );
  const [url, setUrl] = useState("");
  const [notValidUrl, setNotValidUrl] = useState<boolean>(false);
  const uploadRef = useRef<HTMLInputElement | null>(null);
  const [modelResponse, setModelResponse] = useState<ModelResponse | null>(null)
  const [fileId, setFileId] = useState("");

  const handleOption = (opt: "news" | "deepfake") => {
    setSelectedOption(opt);
  };

  const handleAfterUpload = async () => {
    try {
      console.log("FILED ID UPLOADED", fileId)
      setCurrentState(1)
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setCurrentState(2);
      const res = await axios.post(`/api/fake`, {
        videoId: fileId
      });
      setCurrentState(3);
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setCurrentState(4);
      const deletingVideo = await axios.post('/api/delete/video', {
        fileId: fileId
      })
      if(deletingVideo){
        setCurrentState(5)
      }
      const { isFake, confidence, videoId } = res.data;
      setCurrentState(6);
      setModelResponse({
        isFake,
        confidence,
        videoId
      })
      await new Promise((resolve) => setTimeout(resolve, 2000));
      setCurrentState(7)
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setMultiStepLoading(false)
    } catch (error) {
      console.log(error)
    }
  };
  useEffect(() => {
    if (fileId) {
      handleAfterUpload();
    }
  }, [fileId]);


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
        <div className="space-x-6">
          <CustomButton
            isLoading={false}
            Icon={Globe}
            className={cn(
              "rounded-3xl border border-neutral-700 bg-transparent",
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
          <CustomButton
            isLoading={false}
            Icon={Video}
            className={cn(
              "rounded-3xl border border-neutral-700 bg-transparent",
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
        </div>
        <div className="transition-all duration-400">
          {selectedOption === "news" ? (
            <div className="flex h-[40rem] flex-col items-center px-4 pt-10 transition-all duration-400">
              <h2 className="mb-10 text-center text-5xl text-white">
                detect any news blog
              </h2>
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
          ) : (
            <div className="mt-12 flex flex-col items-center transition-discrete duration-400">
              <IKUpload
                className="hidden"
                ref={uploadRef}
                validateFile={(file) => {
                  console.log("THIS IS FILE", file);
                  if (file.size > 1024 * 1024 * 100) {
                    // file should less than 100MB
                    toast.error("File should less than 100mb", {
                      duration: 2000,
                    });
                    return false;
                  }
                  const allowedTypes = ["video/mp4"];
                  if (!allowedTypes.includes(file.type)) {
                    toast.error("Only mp4 are allowed", {
                      duration: 2000,
                    });
                    return false;
                  }
                  return true;
                }}
                onSuccess={(res) => {
                  setCurrentState(1);
                  setFileId(res.fileId)
                }}
                onUploadProgress={(progress) => {
                  console.log("PROGRESS", progress);
                }}
                onUploadStart={(det) => {
                  setMultiStepLoading(true);
                  console.log("ON START", det);
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
            </div>
          )}
        </div>
        {modelResponse !== null ? (
          <div>{JSON.stringify(modelResponse)}</div>
        ): ("")
        }
      </div>
    </div>
  );
}
