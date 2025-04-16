"use server";

import { getServerSession } from "next-auth";
import { authOptions } from "../lib/auth";
import { prisma } from "@repo/db/index";

export async function GetPrismaChats() {
  try {
    const session = await getServerSession(authOptions);

    if (!session?.user) {
      return false;
    }
    const dbRes = await prisma.chats.findMany({
      where: {
        userId: session.user.id,
      },
    });
    if (dbRes) {
      return JSON.stringify(dbRes);
    }
    return false;
  } catch (error) {
    console.log(error);
    return false;
  }
}
