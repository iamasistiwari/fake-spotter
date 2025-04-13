import { NextResponse } from "next/server";
import ImageKit from "imagekit"

const imageKit = new ImageKit({
  publicKey: process.env.NEXT_PUBLIC_KEY!,
  privateKey: process.env.PRIVATE_KEY!,
  urlEndpoint: process.env.NEXT_PUBLIC_URL_ENDPOINT!,
});

export async function GET(){
    return NextResponse.json(imageKit.getAuthenticationParameters())
}