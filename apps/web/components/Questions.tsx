"use client";
import React, { useState } from "react";

const data = [
  {
    title: "What is a deepfake?",
    content:
      "A deepfake is synthetic media—such as video, audio, or images—created using artificial intelligence to mimic real people. It can alter facial expressions, voices, and entire identities, often making it difficult to distinguish from real content.",
  },
  {
    title: "How does your website detect deepfakes?",
    content:
      "Our website uses advanced AI models that analyze visual inconsistencies, audio mismatches, and metadata patterns to identify signs of manipulation. We combine multiple techniques to improve the accuracy of detection across videos, audio, and images.",
  },
  {
    title: "Is the detection 100% accurate?",
    content:
      "While our models are highly trained and accurate, no system can guarantee 100% accuracy. We continuously improve detection capabilities, but some sophisticated deepfakes may still be challenging to identify. It's always good to stay cautious and cross-verify content.",
  },
  {
    title: "What types of media can I check on this platform?",
    content:
      "You can upload or link videos, audio clips, or images. Our platform supports deepfake detection for all three media types, giving you a comprehensive tool to verify authenticity.",
  },
  {
    title: "Is my uploaded data safe?",
    content:
      "Yes. We take privacy seriously. Any files you upload are processed securely and deleted after analysis. Your data is never stored or shared without your consent.",
  },
];

export default function Questions() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggle = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div className="w-full">
      <h2 className="mb-8 text-center text-3xl font-bold">
        Frequently Asked Questions
      </h2>
      <div className="space-y-4">
        {data.map((item, index) => (
          <div
            key={index}
            className="overflow-hidden rounded-2xl border border-gray-300 shadow-sm transition-all duration-200 dark:border-gray-700"
          >
            <button
              onClick={() => toggle(index)}
              className="flex w-full items-center justify-between bg-white px-6 py-4 transition-all duration-200 hover:cursor-pointer hover:bg-gray-100 dark:bg-gray-900 dark:hover:bg-gray-800"
            >
              <span className="text-left text-lg font-medium text-gray-800 transition-all duration-200 dark:text-gray-200">
                {item.title}
              </span>
              <svg
                className={`h-5 w-5 text-gray-500 transition-all duration-300 ${
                  openIndex === index ? "rotate-180" : ""
                }`}
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </button>
            {openIndex === index && (
              <div className="px-6 pt-4 pb-5 text-normal leading-relaxed text-gray-600 transition-all duration-1000 dark:text-gray-300">
                {item.content}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
