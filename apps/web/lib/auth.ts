/* eslint-disable @typescript-eslint/no-unused-vars */
import { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import { prisma } from "@repo/db/index";
import bcrypt from "bcrypt";
import { JWT } from "next-auth/jwt";
import jwt from "jsonwebtoken";
import GoogleProvider from "next-auth/providers/google";
import { Credentials } from "../types/next-auth";

export const authOptions: NextAuthOptions = {
  session: {
    strategy: "jwt",
  },
  pages: {
    signIn: "/signin",
    signOut: "/dashboard",
  },

  secret: process.env.NEXTAUTH_SECRET,
  providers: [
    CredentialsProvider({
      type: "credentials",
      name: "Credentials",
      credentials: {
        name: { label: "Name", type: "text", placeholder: "John Doe" },
        email: {
          label: "Email",
          type: "text",
          placeholder: "john@example.com",
        },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials: Credentials | undefined) {
        if (!credentials) {
          throw new Error("Provide Credentials");
        }
        const { name, email, password } = credentials;
        if (!email || !password) {
          throw new Error("Provide Login Credentials");
        }

        const hashedPassword = await bcrypt.hash(credentials.password, 10);

        const existingUser = await prisma.user.findFirst({
          where: {
            email,
          },
        });
        if (existingUser && existingUser.password && existingUser.provider !== "google") {
          const passwordValidation = await bcrypt.compare(
            password,
            existingUser.password,
          );

          if (passwordValidation) {
            return {
              id: existingUser.id.toString(),
              name: existingUser.name,
              email: existingUser.email,
            };
          }
          throw new Error("Incorrect Password");
        } else {
          if (!name) {
            throw new Error("User Not Found");
          }
          try {
            const user = await prisma.user.create({
              data: {
                email,
                name,
                password: hashedPassword,
                provider: "credential",
              },
            });
            return {
              id: user.id.toString(),
              name: user.name,
              email: user.email,
            };
          } catch (e) {
            console.log(e);
            throw new Error("Error");
          }
        }
      },
    }),
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  callbacks: {
    async jwt({ token, user, account }) {
      if (user) {
        if (account?.provider === "credentials") {
          token.id = user.id;
          token.name = user.name;
          token.email = user.email;
          return token;
        }
        token.id = user.id;
        token.name = user.name;
        token.email = user.email;
        token.picture = user.image;
        return token;
      }
      return token;
    },

    async session({ session, token }) {
      if (token) {
        session.user.id = token.id;
        session.user.name = token.name;
        session.user.email = token.email;
      }
      return session;
    },
    async signIn({ user, account }) {
      try {
        if (
          account?.provider === "google" &&
          user &&
          user.id &&
          user.email &&
          user.name
        ) {
          await prisma.user.upsert({
            where: { id: user.id },
            update: {
              email: user.email,
              name: user.name,
            },
            create: {
              id: user.id,
              email: user.email,
              name: user.name,
              provider: "google",
            },
          });
        }
        return true;
      } catch (error) {
        return false;
      }
    },
  },
};
