"use client";

import { TypewriterEffect } from "./ui/typewriter-effect";


export function TypewriterEffectDemo() {
  const words = [
    {
      text: "With",
    },
    {
      text: "Fake Spotter.",
      className: "text-blue-500 dark:text-blue-500",
    },
  ];
  return (
    <div className="flex h-[40rem] flex-col items-center justify-center">
      
      <TypewriterEffect words={words} />
      
    </div>
  );
}
