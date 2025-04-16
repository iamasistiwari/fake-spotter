import { NextRequest, NextResponse } from "next/server";
import ImageKit from "imagekit";
import { getServerSession } from "next-auth";
import { authOptions } from "../../../../lib/auth";

const imageKit = new ImageKit({
  publicKey: process.env.NEXT_PUBLIC_KEY!,
  privateKey: process.env.PRIVATE_KEY!,
  urlEndpoint: process.env.NEXT_PUBLIC_URL_ENDPOINT!,
});

export async function POST(req: NextRequest) {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized req" }, { status: 400 });
  }
  const data = await req.json();
  const fileId = data.fileId;
  if (!fileId) {
    return NextResponse.json({ error: "fileId required" }, { status: 400 });
  }

  try {
    const result = await imageKit.deleteFile(fileId);
    console.log("RESULT IS", result);
    return NextResponse.json({ success: true });
  } catch (err) {
    console.error("Delete failed:", err);
    return NextResponse.json({ error: "Delete failed" }, { status: 400 });
  }
}
