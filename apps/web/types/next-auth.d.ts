import type { User } from "next-auth";
type UserId = string;

declare module "next-auth/jwt" {
  interface JWT {
    id: UserId;
  }
}

declare module "next-auth" {
  interface Session {
    user: User & {
      id: UserId;
    };
  }
}

interface Credentials {
  name?: string;
  email: string;
  password: string;
}
