-- CreateEnum
CREATE TYPE "ChatType" AS ENUM ('deepfake', 'news');

-- CreateTable
CREATE TABLE "Chats" (
    "id" TEXT NOT NULL,
    "type" "ChatType" NOT NULL,
    "modelResponse" TEXT NOT NULL,
    "userId" TEXT NOT NULL,

    CONSTRAINT "Chats_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "Chats" ADD CONSTRAINT "Chats_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
