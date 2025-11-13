import { useDocDomain } from "@/hooks/use-doc-domain";
import { cn } from "@/lib/utils";
import { LuExternalLink } from "react-icons/lu";
import { Link } from "react-router-dom";

const Banner = ({
  children,
  type,
}: {
  children: React.ReactNode;
  type: "info" | "warning" | "error";
}) => {
  const backgroundClass =
    type === "info"
      ? "bg-info"
      : type === "warning"
        ? "bg-warning"
        : "bg-destructive";
  return (
    <div
      id="banner"
      className={cn(
        "flex w-full items-center justify-center p-2",
        backgroundClass,
      )}
    >
      <span className="text-center text-sm">{children}</span>
    </div>
  );
};

export default Banner;
