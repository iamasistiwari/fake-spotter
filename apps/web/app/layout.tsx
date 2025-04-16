import type { Metadata } from "next";
import "./globals.css";
import Providers from "./providers";
import { Geist, Geist_Mono } from "next/font/google";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],

  weight: "variable",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  weight: ["100", "200", "300", "400", "500", "600", "700", "800", "900"],
});
export const metadata: Metadata = {
  title: "TruFake",
  description: "",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} bg-primary text-white antialiased [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:rounded-2xl [&::-webkit-scrollbar-thumb]:bg-neutral-600 [&::-webkit-scrollbar-track]:bg-transparent`}
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
