"use client";
import React, { FormEvent, useState } from "react";
import CustomButton from "../../../components/ui/CustomButton";
import { signIn } from "next-auth/react";
import toast from "react-hot-toast";
import z from "zod"
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function Signup() {
  const [gLoading, setGLoading] = useState<boolean>(false)
  const [fLoading, setFLoading] = useState<boolean>(false);
  const navigate = useRouter()
  const [input, setInput] = useState({
    name: "",
    password: "",
    email: "",
  });

  const isValidEmail = /\S+@\S+\.\S+/.test(input.email);
  const showEmailError = input.email.length > 0 && !isValidEmail;


  const loginWithGoogle = async () => {
    try {
      setGLoading(true)
      const res = await signIn("google", {callbackUrl: '/dashboard'});
      if (!res?.ok && res) {
        toast.error(res.error, { duration: 1000 });
      }
    } catch (error) {
      toast.error("Something went wrong");
      console.log(error)
    }finally{
      setGLoading(false)
    }
  }


  const loginWithCredentials = async (e: FormEvent) => {
    e.preventDefault()
    try {
      setFLoading(true)
      const zodValdation = z.object({
        name: z.string().min(4),
        email: z.string().email(),
        password: z.string().min(4)
      }).safeParse(input)
      if(!zodValdation.success){
        return
      }

      const res = await signIn('credentials', {
        name: input.name,
        email: input.email,
        password: input.password,
        redirect: false
      })
      if(res?.ok){
        navigate.push('/dashboard')
      }
      if(!res?.ok && res){
        toast.error(res.error, {duration: 1000})
      }
    } catch (e) {
      console.log("eror is ",e)
      toast.error("something went wrong")
    }finally{
      setFLoading(false);
    }
  }

  return (
    <div className="bg-back text-ctext flex h-screen items-center justify-center">
      <div className="flex flex-col space-y-10">
        <div className="text-center text-3xl font-semibold">
          Welcome to <span className="text-safe">FakeSpotter</span>
          <div>
            <span className="font-thin text-base">Already a user?</span>
            <Link href={`/signin`} className="font-medium text-base underline hover:opacity-75 ml-2 transition-opacity duration-200">signin</Link>
          </div>
        </div>
        <form className="flex flex-col space-y-4">
          <div className="flex flex-col space-y-1">
            <label className="font-black" htmlFor="name">
              Name
            </label>
            <input
              onChange={(e) => setInput({ ...input, name: e.target.value })}
              value={input.name}
              id="name"
              className="border-bcolor h-11 w-96 rounded-md border pl-2 text-neutral-100 focus:outline-none"
              placeholder="Johe Doe"
            ></input>
            {input.name.length > 0 && input.name.length < 4 ? (
              <p className="text-danger">
                Name must be at least 4 characters long
              </p>
            ) : null}
          </div>
          <div className="flex flex-col space-y-1">
            <label className="font-black" htmlFor="email">
              Email address
            </label>
            <input
            value={input.email}
              id="email"
              onChange={(e) => setInput({ ...input, email: e.target.value })}
              className="border-bcolor h-11 w-96 rounded-md border pl-2 text-neutral-100 focus:outline-none"
              placeholder="john@gmail.com"
            ></input>
            {showEmailError && (
              <p className="text-danger">Please enter a valid email address</p>
            )}
          </div>

          <div className="flex flex-col space-y-1">
            <span className="flex justify-between">
              <label htmlFor="password" className="font-black">
                Password
              </label>
            </span>
            <input
            value={input.password}
              id="password"
              onChange={(e) => setInput({ ...input, password: e.target.value })}
              className="border-bcolor flex h-11 w-96 justify-center rounded-md border pl-2 font-semibold tracking-widest text-neutral-100 focus:outline-none"
              type="password"
            ></input>
            {input.password.length > 0 && input.password.length < 8 ? (
              <p className="text-danger">Invalid password length</p>
            ) : null}
          </div>
          <CustomButton
            isLoading={fLoading}
            className="text-lg"
            Icon={null}
            loaderStyle="mr-2"
            onClick={loginWithCredentials}
            type="submit"
          >
            Sign in
          </CustomButton>
          <div className="mt-8 flex items-center">
            <div className="bg-bcolor h-0.5 w-full"></div>
            <span className="w-96 text-center text-neutral-100">
              Or continue with
            </span>
            <div className="bg-bcolor h-0.5 w-full"></div>
          </div>

          <div className="flex space-x-4">
            <CustomButton
              isLoading={gLoading}
              className="w-full bg-neutral-200 text-lg text-neutral-900"
              Icon={null}
              loaderStyle="mr-2"
              onClick={loginWithGoogle}
            >
              {gLoading ? null : (
                <svg
                  className="mr-2 h-4 w-4"
                  aria-hidden="true"
                  focusable="false"
                  data-prefix="fab"
                  data-icon="github"
                  role="img"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                >
                  <path
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    fill="#4285F4"
                  />
                  <path
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    fill="#34A853"
                  />
                  <path
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                    fill="#FBBC05"
                  />
                  <path
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                    fill="#EA4335"
                  />
                  <path d="M1 1h22v22H1z" fill="none" />
                </svg>
              )}
              Google
            </CustomButton>
          </div>
        </form>
      </div>
    </div>
  );
}
