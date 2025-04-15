import Link from "next/link";
import React from "react";
import TrueFocus from "../components/ui/TrueFocus";
import RotatingText from "../components/ui/RotatingText";
import { ArrowRight } from "lucide-react";
import { News } from "../lib/news";
import { ImageLens } from "../components/ImageLens";
import Questions from "../components/Questions";

export default function Page() {
  return (
    <div className="min-h-screen bg-black">
      <nav className=" flex h-20 items-center justify-between bg-black pt-6 pr-8 pl-24">
        <div className="flex items-center justify-center space-x-12">
          <TrueFocus
            sentence="Fake Spotter"
            manualMode={false}
            blurAmount={5}
            borderColor="red"
            animationDuration={2}
            pauseBetweenAnimations={1}
          />
          <Link
            href={`/login`}
            className="text-neutral-500 transition-colors duration-500 hover:text-neutral-400"
          >
            Scanner
          </Link>
          <Link
            href={`/contact`}
            className="text-neutral-500 transition-colors duration-500 hover:text-neutral-400"
          >
            Repo
          </Link>
        </div>
        <div className="text-ctext flex items-center justify-center space-x-10 font-medium">
          <Link
            className="rounded-3xl bg-gradient-to-r font-[450]  from-orange-500 to-orange-400 px-10 py-2  text-neutral-800 transition-colors duration-700 hover:from-orange-600 hover:to-orange-300"
            href="/signin"
          >
            Signin
          </Link>
        </div>
      </nav>

      {/* Hero section */}
      <section className="group flex min-h-screen justify-center bg-black pt-20 pr-4 pl-24">
        <span className="absolute right-0 -bottom-20 h-96 w-96 rounded-full bg-pink-500 opacity-20 blur-3xl" />

        <div className="z-10 flex flex-col items-center">
          <div className="flex flex-row items-center space-x-4">
            <div className="text-[100px] font-semibold">Detect</div>
            <RotatingText
              texts={["Videos", "Audio", "Messages", "Images"]}
              mainClassName="px-2 bg-orange-500 h-20 min-w-48  items-center overflow-hidden  py-3 justify-center rounded-xl"
              staggerFrom={"last"}
              initial={{ y: "100%" }}
              animate={{ y: 0 }}
              exit={{ y: "-120%" }}
              staggerDuration={0.025}
              splitLevelClassName="font-bold overflow-hidden pb-1 text-4xl text-white"
              transition={{ type: "spring", damping: 40, stiffness: 400 }}
              rotationInterval={2500}
            />
            <div className="text-[100px] font-semibold">Deepfakes</div>
          </div>
          <div className="relative bottom-10 text-[95px] font-semibold">
            be safe secure
          </div>
          <div className="max-w-[400px] text-center text-neutral-300">
            Take control of deepfake threats with comprehensive detection of
            synthetic audio, video, and images, deployed securely within your
            infastructure.
          </div>
          <Link
            className="mt-16 rounded-3xl bg-gradient-to-r from-orange-500 to-orange-400 px-14 py-3 font-sans text-neutral-800 transition-colors duration-700 hover:from-orange-600 hover:to-orange-300"
            href={"/signup"}
          >
            Try FakeSpotter
          </Link>
        </div>
      </section>

      {/* feature section */}
      <section className="min-h-96 bg-black py-4">
        <div className="flex w-full justify-between pr-8 pl-24">
          <div className="relative z-10 flex w-[40vw] flex-col space-y-8 pt-20">
            {/* Glowing circle in the background */}
            <span className="absolute -top-30 -left-10 z-0 h-48 w-48 rounded-full bg-orange-500 opacity-70 blur-3xl" />

            <span className="z-10 text-4xl font-semibold text-white">
              Why it&apos;s harder to spot a deepfake video?
            </span>
            <span className="text-white">
              It&apos;s harder to spot a deepfake video because AI can
              realistically mimic facial expressions, voice, and movements,
              making the fake content look and sound very convincing.
            </span>
          </div>

          <div className="relative z-50 h-96 w-[650px] overflow-hidden rounded-xl transition-opacity duration-500 hover:cursor-pointer hover:opacity-75">
            <video
              src="/background.mp4"
              autoPlay
              playsInline
              muted
              loop
              className="z-50 rounded-xl object-cover"
            />
          </div>
        </div>
      </section>

      {/* another */}
      <section className="relative min-h-96 border-b border-neutral-900">
        <div className="group flex w-full justify-between pt-40 pr-8 pl-24">
          <div className="flex w-[40vw] flex-col space-y-8">
            <span className="z-50 text-5xl font-thin">
              2024 Enterprise Deepfake Threat Report: Analysis of Risks and
              Mitigation Strategies
            </span>
          </div>
          <div className="z-20 flex h-96 w-[650px] flex-col space-y-10">
            <span className="z-50">
              A comprehensive analysis of how synthetic media and deepfakes are
              reshaping enterprise security in 2024. Based on extensive research
              and real-world case studies, this report examines emerging threat
              vectors, quantifies business impact, and provides actionable
              frameworks for protecting your organization.
            </span>
            <Link
              href={"/"}
              className="group text-orange-400 text-saf flex h-20 max-w-72 items-center space-x-1 transition-colors duration-200 hover:cursor-pointer hover:text-orange-500"
            >
              <span className="">Download full report</span>
              <ArrowRight className=" size-6 transition-all duration-200 group-hover:ml-1" />
            </Link>
            <span className="absolute top-20 left-[400px] z-0 h-96 w-96 rounded-full bg-pink-900 opacity-30 blur-3xl transition-opacity duration-1000 group-hover:opacity-45" />
          </div>
        </div>
      </section>

      <div className="group relative flex min-h-screen flex-col justify-center space-y-4 bg-black pr-8 pl-24">
        <div className="z-50 pt-10 text-center text-4xl font-semibold">
          Few Cases of AI Deepfake Fraud From 2024 Exposed
        </div>

        <span className="absolute top-1/2 left-1/2 z-0 h-[600px] w-[900px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-green-500 opacity-10 blur-[380px] transition-opacity duration-1000 group-hover:opacity-50" />

        <div className="z-50 flex items-center justify-center space-x-12">
          {News.map((news, index) => (
            <ImageLens
              key={index}
              title={news.title}
              description={news.description}
              imageUrl={news.imageUrl}
            />
          ))}
        </div>
      </div>

      {/* asked questions */}
      <div className="bg-black px-24 pt-32">
        <Questions />
      </div>

      <footer className="mt-20 border-t border-neutral-900 py-6 pt-16 text-gray-400">
        <div className="mx-auto flex max-w-7xl flex-row items-center justify-between px-4">
          <p className="text-sm">
            &copy; {new Date().getFullYear()} Fake Spotter. All rights reserved.
          </p>
          <div className="mt-2 flex space-x-4 sm:mt-0">
            <a
              href="#"
              className="transition-colors hover:text-black dark:hover:text-white"
            >
              Privacy Policy
            </a>
            <a
              href="#"
              className="transition-colors hover:text-black dark:hover:text-white"
            >
              Terms of Service
            </a>
            <a
              href="#"
              className="transition-colors hover:text-black dark:hover:text-white"
            >
              Contact
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
