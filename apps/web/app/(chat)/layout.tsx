import React from "react";
import Sidebar from "../../components/Sidebar";
import Image from "next/image";
import { getServerSession } from "next-auth";
import { authOptions } from "../../lib/auth";
export default async function Layout({ children }: { children: React.ReactNode }) {
    const session = await getServerSession(authOptions);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-[#212121]">
      <Sidebar />
      <main className="flex-1">
        <nav className="flex h-16 items-center justify-between border-b border-neutral-800 px-4">
          <a href="/dashboard" className="text-lg font-semibold hover:cursor-pointer">
            TruFake
          </a>
          <div className="cursor-pointer">
            <div className="relative h-10 w-10 rounded-full ring-neutral-700 transition-discrete duration-700 hover:ring-3">
              <Image
                src={session?.user.image ||"/user.png"}
                alt="pic"
                fill
                className="rounded-full opacity-70"
              />
            </div>
          </div>
        </nav>
        {children}
      </main>
    </div>
  );
}
