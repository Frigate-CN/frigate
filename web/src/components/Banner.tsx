import { useDocDomain } from "@/hooks/use-doc-domain";
import { LuExternalLink } from "react-icons/lu";
import { Link } from "react-router-dom";

const Banner = ({
  children,
  type,
  className
}: {
  children: React.ReactNode;
  type: "info" | "warning" | "error";
  className?: string;
}) => {
  const { getLocaleDocUrl } = useDocDomain();
  return (
    <div
      id="banner"
      className="flex w-full items-center justify-center bg-destructive p-2"
    >
      <span className="text-center text-sm">
        You are currently accessing the system through an unprotected port
        without authentication, which may pose a serious risk of data leakage.
        Please ensure this port is inaccessible from external networks. <br />
        <Link
          to={getLocaleDocUrl("configuration/live")}
          target="_blank"
          rel="noopener noreferrer"
          className="inline"
        >
          {t("readTheDocumentation", { ns: "common" })}
          <LuExternalLink className="ml-2 inline-flex size-3" />
        </Link>
      </span>
    </div>
  );
};
