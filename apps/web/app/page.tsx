import Image from "next/image";
import Link from "next/link";
import React from "react";
import CircularText from "../components/ui/CircularText";
import TrueFocus from "../components/ui/TrueFocus";
import RotatingText from "../components/ui/RotatingText";
import GradientText from "../components/ui/GradientText";
import { TypewriterEffect } from "../components/ui/typewriter-effect";
import { ArrowRight, Github } from "lucide-react";
import { News } from "../lib/news";
import { ImageLens } from "../components/ImageLens";
import Questions from "../components/Questions";

export default function Page() {

  const words = [
    {
      text: "With",
      className: "text-neutral-400 dark:text-neutral-400",
    },
    {
      text: "Fake Spotter.",
      className: "text-ctext dark:text-ctext",
    },
  ];
  return (
    <div className="bg-back min-h-screen">
      <nav className="flex h-16 items-center justify-between border-b border-bcolor pr-8 pl-24">
        <div className="">
          <TrueFocus
            sentence="Fake Spotter"
            manualMode={false}
            blurAmount={5}
            borderColor="red"
            animationDuration={2}
            pauseBetweenAnimations={1}
          />
        </div>
        <div className="text-ctext flex items-center justify-center space-x-10 font-medium">
          <Link href={`/contact`} className="">
            Scanner
          </Link>
          <Link href={`/contact`}>About</Link>
          <Link href={`/contact`}>Contact</Link>
          <Link
            className="border-bcolor hover:bg-sec rounded-md border px-6 py-2 font-semibold transition-all duration-200 hover:mb-1"
            href={`/signin`}
          >
            SIGN IN
          </Link>
        </div>
      </nav>

      {/* Hero section */}
      <section className="mt-20 flex min-h-screen justify-between pr-4 pl-24">
        <div className="flex flex-col">
          <div className="mt-20 flex w-full items-center justify-between text-start">
            <div>
              <GradientText
                colors={[
                  "#1E3A8A",
                  "#3B82F6",
                  "#10B981",
                 
                ]}
                animationSpeed={10}
                showBorder={false}
                className="text-start text-5xl font-semibold"
              >
                Detect deepfakes
              </GradientText>
            </div>
            <RotatingText
              texts={["Videos", "Audio", "Messages", "Images"]}
              mainClassName="px-2 bg-safe min-w-40  overflow-hidden text-neutral-100 py-3 justify-center rounded-md"
              staggerFrom={"last"}
              initial={{ y: "100%" }}
              animate={{ y: 0 }}
              exit={{ y: "-120%" }}
              staggerDuration={0.025}
              splitLevelClassName="font-semibold overflow-hidden pb-1 text-2xl text-ctext"
              transition={{ type: "spring", damping: 40, stiffness: 400 }}
              rotationInterval={2500}
            />
          </div>
          <div className="mt-4">
            <TypewriterEffect words={words} className="text-start" />
          </div>
          <div className="mt-20 w-[40vw] text-xl font-thin tracking-wider text-neutral-300 transition-opacity duration-200 select-none hover:opacity-75">
            <p>
              Take control of deepfake threats with comprehensive detection of
              synthetic audio, video, and images, deployed securely within your
              infastructure.
            </p>
          </div>
          <div className="mt-10 flex space-x-8">
            <Link
              href={`https://github.com/iamasistiwari/fake-spotter`}
              target="_blank"
              className="group relative shadow-lg shadow-surf flex items-center justify-center rounded-md border border-bcolor bg-transparent px-8 py-2 transition-all duration-200 hover:cursor-pointer hover:opacity-75"
            >
              <Github className="mr-1 size-5" />
              <span className="absolute top-2 right-2 h-2 w-2 bg-neutral-300 transition-all group-hover:rotate-45"></span>
              Open Source
            </Link>
            <Link
              href={`/signup`}
              className="group relative shadow-lg shadow-surf font-semibold flex items-center justify-center rounded-md bg-safe px-8 py-1 text-white transition-opacity duration-200 hover:cursor-pointer hover:opacity-75"
            >
              Get Started
            </Link>
          </div>
        </div>

        <div className="relative h-96 w-[650px] overflow-hidden rounded-4xl shadow-md hover:shadow-lg transition-shadow duration-300  shadow-safe">
          <Image src="/img2.webp" alt="img1" fill className="object-cover" />

          <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-neutral-100 via-transparent to-transparent" />
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-tr from-neutral-100 via-transparent to-transparent" />

          <CircularText
            text="Deepfake*Detected*"
            onHover="speedUp"
            spinDuration={20}
            className="absolute right-0 z-20 text-black"
          />
        </div>
      </section>

      {/* feature section */}
      <section className="min-h-96 border-t border-bcolor bg-bcolor py-4">
        <div className="flex w-full justify-between pr-8 pl-24">
          <div className="flex w-[40vw] flex-col space-y-8 pt-20">
            <span className="text-4xl font-semibold">
              Why it&apos;s harder to spot a deepfake video ?
            </span>
            <span>
              It&apos;s harder to spot a deepfake video because AI can
              realistically mimic facial expressions, voice, and movements,
              making the fake content look and sound very convincing.
            </span>
          </div>
          <div className="relative h-96 w-[650px] overflow-hidden rounded-4xl border border-neutral-700 transition-opacity duration-200 hover:cursor-pointer hover:opacity-75">
            <Image src="/img3.jpg" alt="img1" fill className="object-cover" />
          </div>
        </div>
      </section>

      {/* another */}
      <section className="min-h-96 border-b border-neutral-800 bg-[#141518]">
        <div className="flex w-full justify-between pt-40 pr-8 pl-24">
          <div className="flex w-[40vw] flex-col space-y-8">
            <span className="text-5xl font-thin">
              2024 Enterprise Deepfake Threat Report: Analysis of Risks and
              Mitigation Strategies
            </span>
          </div>
          <div className="flex h-96 w-[650px] flex-col space-y-10">
            <span>
              A comprehensive analysis of how synthetic media and deepfakes are
              reshaping enterprise security in 2024. Based on extensive research
              and real-world case studies, this report examines emerging threat
              vectors, quantifies business impact, and provides actionable
              frameworks for protecting your organization.
            </span>
            <Link
              href={"/"}
              className="group text-safe flex h-20 max-w-72 items-center space-x-1 text-saf transition-colors duration-200 hover:cursor-pointer hover:text-green-700"
            >
              <span className="">Download full report</span>
              <ArrowRight className="size-6 transition-all duration-200 group-hover:ml-1" />
            </Link>
          </div>
        </div>
      </section>

      <div className="flex min-h-screen flex-col justify-center space-y-4 pr-8 pl-24">
        <div className="pt-10 text-center text-4xl font-semibold">
          Few Cases of AI Deepfake Fraud From 2024 Exposed
        </div>
        <div className="flex items-center justify-center space-x-12">
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
      <div className="mt-32 px-24">
        <Questions />
      </div>

      <footer className="mt-16 border-bcolor border-t py-6 text-gray-400">
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
