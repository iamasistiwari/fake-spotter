import { NextRequest, NextResponse } from "next/server";
import ImageKit from "imagekit";

const imageKit = new ImageKit({
  publicKey: process.env.NEXT_PUBLIC_KEY!,
  privateKey: process.env.PRIVATE_KEY!,
  urlEndpoint: process.env.NEXT_PUBLIC_URL_ENDPOINT!,
});

export async function POST(req: NextRequest) {
  const { fileId } = await req.json();

  if (!fileId) {
    return NextResponse.json({ error: "fileId required" }, { status: 400 });
  }

  try {
    const result = await imageKit.deleteFile(fileId);
    console.log("RESULT IS", result)
    return NextResponse.json({ success: true });
  } catch (err) {
    console.error("Delete failed:", err);
    return NextResponse.json({ error: "Delete failed" }, { status: 500 });
  }
}
