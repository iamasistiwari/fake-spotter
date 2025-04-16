import { getToken } from "next-auth/jwt";
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export async function middleware(request: NextRequest) {
  const secret = process.env.NEXTAUTH_SECRET;
  const token = await getToken({ req: request, secret });

  const { pathname } = request.nextUrl;

  if (
    !token &&
    (pathname.startsWith("/dashboard") || pathname.startsWith("/signout"))
  ) {
    return NextResponse.redirect(new URL("/", request.url));
  }

  if (token && pathname === "/") {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/", "/dashboard", "/signout"],
};
