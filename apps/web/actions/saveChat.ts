"use server";

import { getServerSession } from "next-auth";
import { authOptions } from "../lib/auth";
import { prisma } from "@repo/db/index";

export async function SavePrismaChats({
  modelResponse,
  type,
  title,
}: {
  title: string;
  modelResponse: string;
  type: "deepfake" | "news";
}) {
  try {
    const session = await getServerSession(authOptions);

    if (!session?.user) {
      return false;
    }

    // const dbRes = await prisma.chat.create({
    //   data: {
    //     title,
    //     modelResponse,
    //     type,
    //     userId: session.user.id,
    //   },
    // });
    console.log("HERE")
    const dbRes = await prisma.chat.create({
      data: {
        title,
        modelResponse,
        type,
        userId: session.user.id,
      },
    });
    console.log("AFTER")

    console.log("DATABASE", dbRes);

    return !!dbRes;
  } catch (error) {
    console.log(error);
    return false;
  }
}
