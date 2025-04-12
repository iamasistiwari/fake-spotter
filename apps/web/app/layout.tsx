import type { Metadata } from "next";
import { Special_Elite } from "next/font/google";
import "./globals.css";
import Providers from "./providers";

const specialGothic = Special_Elite({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-special-gothic",
});


export const metadata: Metadata = {
  title: "Ai-App",
  description: "",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${specialGothic.variable} bg-primary text-white antialiased`}
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
