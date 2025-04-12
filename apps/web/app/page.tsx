import Image from "next/image";
import Link from "next/link";
import React from "react";

export default function page() {
  return (
    <div className="bg-white min-h-screen text-black">
      <nav className="h-16 border-b border-neutral-300 justify-between items-center flex px-24">
        <div className="">
          <span className="text-xl">Fake Spotter</span>
        </div>
        <div className="flex space-x-10 font-medium text-neutral-600 items-center justify-center">
          <Link href={`/contact`}>Scanner</Link>
          <Link href={`/contact`}>About</Link>
          <Link href={`/contact`}>Contact</Link>
          <Link
            className="border py-2 rounded-md font-normal hover:mb-1 text-black hover:text-blue-500 border-neutral-400 transition-all duration-200 px-6"
            href={`/contact`}
          >
            SIGN IN
          </Link>
        </div>
      </nav>

      {/* Hero section */}
      <section className="min-h-screen flex justify-between pl-24 pr-4 mt-10">
        <div>
          <span>Detect deepfakes before they exploit</span>
        </div>

        <div className="relative w-[650px] h-96 rounded-4xl overflow-hidden">
          <Image src="/img2.webp" alt="img1" fill className="object-cover" />

          {/* Gradient overlay at the corner */}
          <div className="absolute inset-0 bg-gradient-to-br from-neutral-100 via-transparent to-transparent pointer-events-none" />
          {/* <div className="absolute  inset-0 bg-gradient-to-bl from-neutral-100 via-transparent to-transparent pointer-events-none" /> */}
          <div className="absolute  inset-0 bg-gradient-to-tr from-neutral-100 via-transparent to-transparent pointer-events-none" />

          {/* <div className="absolute  inset-0 bg-gradient-to-tl from-neutral-100 via-transparent to-transparent pointer-events-none" /> */}
        </div>
      </section>
    </div>
  );
}
