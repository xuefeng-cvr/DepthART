const envRepoName = process.env.NEXT_PUBLIC_REPO_NAME ?? "DepthART";

const resolveBasePath = () => {
	if (envRepoName) return `/${envRepoName}`;

	if (typeof window === "undefined") {
		return "";
	}

	const firstSegment = window.location.pathname.split("/")[1] || "";
	return firstSegment ? `/${firstSegment}` : "";
};

export const basePath = resolveBasePath();
export const withBasePath = (path: string) => `${basePath}${path}`;
