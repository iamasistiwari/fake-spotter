"use server";

import { getServerSession } from "next-auth";
import { authOptions } from "../lib/auth";
import { prisma } from "@repo/db/index";

export async function SavePrismaChats({
  modelResponse,
  type,
}: {
  modelResponse: string;
  type: "deepfake" | "news";
}) {
  try {
    const session = await getServerSession(authOptions);

    if (!session?.user) {
      return false;
    }
    const dbRes = await prisma.chats.create({
      data: {
        modelResponse,
        type,
        userId: session.user.id,
      },
    });

    if (dbRes) {
      return true;
    }
    return false;
  } catch (error) {
    console.log(error);
    return false;
  }
}
